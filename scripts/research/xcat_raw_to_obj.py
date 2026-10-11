#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_raw(path: Path) -> Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]]:
    organs: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]] = {}
    current_label: str | None = None
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []

    def save_current() -> None:
        nonlocal triangles, current_label
        if current_label and triangles:
            organs[current_label] = list(triangles)
        triangles = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                values = [float(p) for p in parts]
                is_numeric = True
            except ValueError:
                is_numeric = False

            if not is_numeric:
                save_current()
                current_label = line.replace(" ", "_")
                continue

            if len(values) % 9 != 0:
                raise ValueError(f"Invalid numeric row length in {path.name}: {len(values)}")

            for i in range(0, len(values), 9):
                tri = values[i:i + 9]
                triangles.append(
                    (
                        (tri[0], tri[1], tri[2]),
                        (tri[3], tri[4], tri[5]),
                        (tri[6], tri[7], tri[8]),
                    )
                )

    save_current()
    return organs


def write_obj(organs: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]], output_path: Path) -> None:
    vertex_index = 1
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"o {output_path.stem}\n")
        for label, tris in organs.items():
            f.write(f"g {label}\n")
            for tri in tris:
                for v in tri:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                f.write(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n")
                vertex_index += 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XCAT .raw surface files to OBJ.")
    parser.add_argument("raw_file", type=Path, help="Input .raw file")
    parser.add_argument("--out", type=Path, default=None, help="Output .obj path (default: same name)")
    args = parser.parse_args()

    raw_path = args.raw_file
    if not raw_path.exists():
        raise SystemExit(f"Input file not found: {raw_path}")

    out_path = args.out if args.out else raw_path.with_suffix(".obj")
    organs = parse_raw(raw_path)
    write_obj(organs, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
