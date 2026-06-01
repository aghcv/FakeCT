#!/bin/bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/xcat_pool.sh --phantom_id <id> [options]

Required:
  --phantom_id <id>          Base phantom id for job naming and outputs

Options:
  --set <key=val1,val2>      Parameter sweep (repeatable; comma-separated values)
  --organ_file <file>        Shortcut for --set organ_file=<file>
  --heart_base <file>        Shortcut for --set heart_base=<file>
  --base_dir <dir>           Base directory for binary/params/outputs (default: ./outputs/xcat)
  --template <file>          Parameter template (default: <base_dir>/general.samp.par)
  --binary <file>            XCAT binary (default: <base_dir>/dxcat2_linux_64bit)
  --sbatch_args <args>       Extra sbatch args (default: empty)
  --dry_run                  Print commands without submitting
USAGE
}

phantom_id=""
param_keys=()
param_values=()
base_dir="./outputs/xcat"
template=""
binary=""
sbatch_args=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phantom_id)
      phantom_id="$2"
      shift 2
      ;;
    --set)
      if [[ "$2" != *=* ]]; then
        echo "Invalid --set format (expected key=val1,val2): $2" >&2
        exit 1
      fi
      param_keys+=("${2%%=*}")
      param_values+=("${2#*=}")
      shift 2
      ;;
    --organ_file)
      param_keys+=("organ_file")
      param_values+=("$2")
      shift 2
      ;;
    --heart_base)
      param_keys+=("heart_base")
      param_values+=("$2")
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
    --sbatch_args)
      sbatch_args="$2"
      shift 2
      ;;
    --dry_run)
      dry_run=1
      shift 1
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

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH" >&2
  exit 1
fi

read -r -a sbatch_args_arr <<< "$sbatch_args"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

split_values() {
  local raw="$1"
  IFS=',' read -r -a values <<< "$raw"
  for i in "${!values[@]}"; do
    values[$i]="$(trim "${values[$i]}")"
  done
  printf '%s\n' "${values[@]}"
}

sanitize_id() {
  printf '%s' "$1" | tr -c '[:alnum:]_.-' '_'
}

total_combos=1
for i in "${!param_keys[@]}"; do
  list=$(split_values "${param_values[$i]}")
  count=$(printf '%s\n' "$list" | grep -c '.*')
  if [[ $count -eq 0 ]]; then
    count=1
  fi
  total_combos=$((total_combos * count))
done

submit_job() {
  local job_id="$1"
  shift
  local -a set_args=("$@")
  local -a cmd=(sbatch "${sbatch_args_arr[@]}" --job-name="$job_id" scripts/xcat_job.sh \
    --phantom_id "$job_id" \
    --base_dir "$base_dir" \
    --template "$template" \
    --binary "$binary")

  if [[ ${#set_args[@]} -gt 0 ]]; then
    cmd+=("${set_args[@]}")
  fi

  if [[ $dry_run -eq 1 ]]; then
    printf 'DRY RUN: %q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

build_jobs() {
  local idx="$1"
  shift
  local -a current_sets=("$@")

  if [[ $idx -ge ${#param_keys[@]} ]]; then
    local job_id="$phantom_id"
    if [[ $total_combos -gt 1 ]]; then
      local suffix=""
      for i in "${!current_sets[@]}"; do
        if [[ "${current_sets[$i]}" == --set ]]; then
          local kv="${current_sets[$((i + 1))]}"
          suffix+="__$(sanitize_id "$kv")"
        fi
      done
      job_id+="$suffix"
    fi
    submit_job "$job_id" "${current_sets[@]}"
    return
  fi

  local key="${param_keys[$idx]}"
  local values
  IFS=$'\n' read -r -d '' -a values < <(split_values "${param_values[$idx]}") || true

  if [[ ${#values[@]} -eq 0 ]]; then
    values=("")
  fi

  for value in "${values[@]}"; do
    if [[ -n "$value" ]]; then
      build_jobs $((idx + 1)) "${current_sets[@]}" --set "${key}=${value}"
    else
      build_jobs $((idx + 1)) "${current_sets[@]}"
    fi
  done
}

build_jobs 0
