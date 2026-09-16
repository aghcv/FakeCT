#!/bin/bash
#SBATCH --job-name=xcat
#SBATCH --output=logs/xcat_%x.%j.out
#SBATCH --error=logs/xcat_%x.%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sbatch scripts/xcat_job.sh --phantom_id <id> [options]

Required:
  --phantom_id <id>          Phantom id used for job naming and outputs

Options:
  --set <key=value>          Override any parameter (repeatable)
  --organ_file <file>        Shortcut for --set organ_file=<file>
  --heart_base <file>        Shortcut for --set heart_base=<file>
  --base_dir <dir>           Base directory for binary/params/outputs (default: ./outputs/xcat)
  --template <file>          Parameter template (default: <base_dir>/general.samp.par)
  --binary <file>            XCAT binary (default: <base_dir>/dxcat2_linux_64bit)
  --output_dir <dir>         Output directory (default: <base_dir>/<phantom_id>)
  --convert_raw <0|1>        Convert *.raw to OBJ after run (default: 1)
USAGE
}

phantom_id=""
set_keys=()
set_values=()
base_dir="./outputs/xcat"
template=""
binary=""
output_dir=""
convert_raw=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phantom_id)
      phantom_id="$2"
      shift 2
      ;;
    --set)
      if [[ "$2" != *=* ]]; then
        echo "Invalid --set format (expected key=value): $2" >&2
        exit 1
      fi
      set_keys+=("${2%%=*}")
      set_values+=("${2#*=}")
      shift 2
      ;;
    --organ_file)
      set_keys+=("organ_file")
      set_values+=("$2")
      shift 2
      ;;
    --heart_base)
      set_keys+=("heart_base")
      set_values+=("$2")
      shift 2
      ;;
    --base_dir)
      base_dir="$2"
      shift 2
      ;;
    --template)
      template="$2"
      shift 2
      ;;
    --binary)
      binary="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --convert_raw)
      convert_raw="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac

done

if [[ -z "$phantom_id" ]]; then
  echo "Missing --phantom_id" >&2
  usage
  exit 1
fi

if [[ -z "$template" ]]; then
  template="${base_dir}/general.samp.par"
fi

if [[ -z "$binary" ]]; then
  binary="${base_dir}/dxcat2_linux_64bit"
fi

if [[ -z "$output_dir" ]]; then
  output_dir="${base_dir}/${phantom_id}"
fi

param_file="${base_dir}/${phantom_id}.par"

mkdir -p logs
mkdir -p "$output_dir"

if [[ ! -f "$template" ]]; then
  echo "Template not found: $template" >&2
  exit 1
fi

if [[ ! -x "$binary" ]]; then
  echo "Binary not found or not executable: $binary" >&2
  exit 1
fi

cp "$template" "$param_file"

tmp_file="$(mktemp)"

escape_regex() {
  printf '%s' "$1" | sed -e 's/[][\/.*^$]/\\&/g'
}

for i in "${!set_keys[@]}"; do
  key="${set_keys[$i]}"
  value="${set_values[$i]}"
  key_re="$(escape_regex "$key")"

  if grep -Eq "^[[:space:]]*${key_re}[[:space:]]*=" "$param_file"; then
    sed -E "s/^[[:space:]]*${key_re}[[:space:]]*=.*/${key} = ${value}/" "$param_file" > "$tmp_file"
    mv "$tmp_file" "$param_file"
  else
    printf '\n%s = %s\n' "$key" "$value" >> "$param_file"
  fi
done

"$binary" "$param_file" "$output_dir"

if [[ "$convert_raw" == "1" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  converter="${script_dir}/xcat_raw_to_obj.py"
  if [[ -f "$converter" ]]; then
    shopt -s nullglob
    raw_files=("${output_dir}"/*.raw)
    for raw_file in "${raw_files[@]}"; do
      python3 "$converter" "$raw_file"
    done
  else
    echo "Converter not found: $converter" >&2
  fi
fi
