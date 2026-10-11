#!/usr/bin/env python3
"""Bundle the saved two-model reports for self-contained local assessment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="User-facing date, YYYY-MM-DD")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise ValueError("Package date must use YYYY-MM-DD")
    study = ROOT / "outputs/experiments/aorta/pretrained-finetune-r1"
    assets = study / "two-arm-ct-r1"
    exports = ROOT / "outputs/exports"; exports.mkdir(exist_ok=True)
    name = "fakect-aorta-pretrained-vs-finetuned-" + args.date
    target = exports / (name + ".zip")
    if target.exists(): raise FileExistsError("Preserve published ZIPs; use a new revision")
    sources = {"comparison": study / "review-pretrained-finetuned-r1",
               "narrowing-stress": assets / "stress", "young-child-diagnostic": assets / "young-child"}
    for source in sources.values():
        provenance = json.loads((source / "provenance.json").read_text())
        for relative, expected in provenance["output_sha256"].items():
            if sha(source / relative) != expected: raise ValueError("Source changed: " + relative)
    with tempfile.TemporaryDirectory(prefix="fakect-two-model-review-") as temporary:
        bundle = Path(temporary) / name; bundle.mkdir()
        for directory, source in sources.items(): shutil.copytree(source, bundle / directory)
        methods = bundle / "methods"; methods.mkdir()
        for filename in ("input.ini", "resolution-review.json", "resolution-review.json.sha256",
                "study-lock.json", "study-lock.json.sha256", "training-completion.json", "training-completion.json.sha256"):
            shutil.copy2(study / filename, methods / filename)
        shutil.copy2(ROOT / "requirements-pretrained.txt", methods / "requirements-pretrained.txt")
        shutil.copy2(ROOT / "scripts/evaluation/aorta_two_arm_ct.py", methods / "aorta_two_arm_ct.py")
        (methods / "STUDY.txt").write_text('''Pretrained versus Fine-tuned — displayed methods

Pretrained: unchanged TotalSegmentator 2.18.0 cardiac nnU-Net weights.
Fine-tuned: the same network adapted with original phantoms plus digital
siblings. The displayed predictions come from the sibling-trained arm.

Five family-held-out folds use eight TRAIN and two held-out original phantoms.
Siblings stay with their source phantom. Each TRAIN case supplies its original
plus sixteen siblings. Held-out cases supply an original plus twelve other
burden-spanning siblings: 130 scans across folds, including 120 modified cases.
Fine-tuning receives 2,000 batch-one 128-cubed patch updates per fold, with
1.5 mm model sampling. CT is the model input. Preprocessing, supplied atlas
region, inference and native-grid evaluation are identical in both models.

Saved measurements and native CT planes/coordinates/windows are unchanged.
Whole-aorta mean Dice is 0.71894 Pretrained / 0.90631 Fine-tuned; change-local
Dice is 0.68671 / 0.82666. Local surface Dice at 1 mm is 0.65460 / 0.84110.
Local HD95 has 112/120 and 117/120 valid surfaces, respectively. Undefined
empty predictions remain failures in Dice/recall/surface Dice.

Histograms cover all 2,560 generated siblings. Mixed changes appear in both
addition and removal plots; 39 siblings have no net aorta-core edit. Counts
are synthetic burden rather than validated clinical severity. Native voxel
sizes differ; per-phantom and physical-volume views provide context.

Three main CT examples were frozen before inference. Additional stress and
young-child examples are explicitly post-hoc. Every panel has Reference /
Pretrained / Fine-tuned columns; native geometry is preserved. The 3D viewer
retains original meshes and independent visibility/opacity controls.

This is a region-assisted phantom development study. Siblings are correlated;
130 scans are not 130 independent patients. Frozen source configuration,
metadata and job receipts are included for traceability.
''', encoding="utf-8")
        (bundle / "README.txt").write_text('''FakeCT — Pretrained versus Fine-tuned

Extract the ZIP completely and open index.html in a desktop browser.
Keep the folder layout intact. The 3D viewer uses bundled JavaScript/Plotly
and requires WebGL; its standalone page is linked from the home page.

Fine-tuned means adaptation using original phantoms plus digital siblings.
CT panels have three columns: Reference / Pretrained / Fine-tuned.
Cyan contours: current-scan reference; magenta contours: prediction.
Performance curves: cyan Pretrained, magenta Fine-tuned.

comparison/ contains the report, offline 3D and PNG/PDF/SVG figures.
narrowing-stress/ and young-child-diagnostic/ contain post-hoc illustrations.
methods/ contains study notes, frozen source configuration and metadata.

Both displayed models have predictions for the same 130 held-out scans.
CSV has 260 model/scan records. Measurements and native CT geometry are
unchanged. Source provenance retains original Wahab paths; package checksums
cover the files needed for self-contained local viewing.
''', encoding="utf-8")
        document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pretrained versus Fine-tuned</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:auto;padding:28px;color:#183144}a{color:#006b88}.card{border:2px solid #dce8ed;border-radius:10px;padding:16px;margin:16px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}</style></head><body><h1>Pretrained versus Fine-tuned</h1><p>FakeCT aorta segmentation · DATE · 130 matched held-out scans</p><p>Fine-tuned means adaptation using original phantoms plus digital siblings.</p><div class="card"><h2><a href="comparison/report.html">Open the final two-model report</a></h2><p>Three-column CT panels: Reference, Pretrained, Fine-tuned. Two-model performance curves, edited-voxel histograms and per-family results.</p><p><a href="comparison/report-3d/index.html">Open interactive 3D directly</a> · <a href="comparison/metrics.csv">Metrics CSV</a> · <a href="comparison/analysis.json">Displayed analysis</a></p></div><div class="card"><h2>Additional CT examples</h2><p><a href="narrowing-stress/index.html">Narrowing stress example</a> — 6,177 removed core voxels; local Dice 0.830 Pretrained and 0.922 Fine-tuned.</p><p><a href="young-child-diagnostic/index.html">Young-child diagnostic</a> — 4,750 removed core voxels; shows remaining local segmentation errors.</p><p>These two examples are post-hoc illustrations. The three main CT examples were selected before inference.</p></div><table><tr><th>Mean Dice</th><th>Pretrained</th><th>Fine-tuned</th></tr><tr><td>Whole aorta · 130 scans</td><td>0.719</td><td>0.906</td></tr><tr><td>Change-local · 120 edited scans</td><td>0.687</td><td>0.827</td></tr></table><p>Cyan: reference CT contours and Pretrained curves. Magenta: prediction CT contours and Fine-tuned curves. Figures have PNG/PDF/SVG versions.</p><p><a href="README.txt">Opening instructions</a> · <a href="methods/STUDY.txt">Study methods</a> · <a href="methods/input.ini">Source configuration</a> · <a href="package-manifest.json">Package checksums</a></p></body></html>'''
        (bundle / "index.html").write_text(document.replace("DATE", args.date), encoding="utf-8")
        shutil.copy2(Path(__file__), methods / "package-generator.py")
        fingerprints = {str(f.relative_to(bundle)): dict(bytes=f.stat().st_size, sha256=sha(f))
                        for f in sorted(bundle.rglob('*')) if f.is_file()}
        # Supplementary pages link to their shared parent's provenance.
        # Give that link package-level provenance with the actual portable paths.
        package_provenance = dict(schema_version="fakect.two-model-package-provenance/1",
            complete=True, source_reports={k: dict(path=str(v), provenance_sha256=sha(v / "provenance.json"))
                for k, v in sources.items()},
            source_ct_assets_provenance_sha256=sha(assets / "provenance.json"),
            models_run=False, metrics_recomputed=False,
            output_sha256={k: v["sha256"] for k, v in fingerprints.items()})
        (bundle / "provenance.json").write_text(json.dumps(package_provenance, indent=2) + "\n")
        fingerprints["provenance.json"] = dict(bytes=(bundle / "provenance.json").stat().st_size,
            sha256=sha(bundle / "provenance.json"))
        manifest = dict(schema_version="fakect.two-model-offline-package/1", date=args.date,
            timezone="America/New_York", entry_point="index.html",
            displayed_models={"pretrained": "Pretrained", "siblings": "Fine-tuned"},
            source_reports={k: str(v) for k, v in sources.items()}, paired_scans=130,
            models_rerun=False, metrics_recomputed=False, files=fingerprints)
        (bundle / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (bundle / "SHA256SUMS.txt").write_text(''.join(v["sha256"] + "  " + k + "\n" for k, v in fingerprints.items()))
        for page in bundle.rglob('*.html'):
            content = page.read_text()
            if re.search(r"Baseline-only|Sibling fine-tuning", content, re.I): raise ValueError("Stale model label: " + str(page))
            for link in re.findall(r'(?:href|src)="([^"]+)"', content):
                if not link.startswith(('http:', 'https:', 'data:', '#')) and not (page.parent / link).exists():
                    raise ValueError("Missing offline target: " + str(page) + " -> " + link)
        for image in bundle.rglob('*.svg'):
            if re.search(r"Baseline-only|Sibling fine-tuning", image.read_text(), re.I): raise ValueError("Stale figure label")
        analysis = json.loads((bundle / "comparison/analysis.json").read_text())
        if set(analysis["arms"]) != {"pretrained", "siblings"} or analysis["paired_samples"] != 130:
            raise ValueError("Unexpected displayed model inventory")
        rows = list(csv.DictReader((bundle / "comparison/metrics.csv").open()))
        if len(rows) != 260 or {r["model"] for r in rows} != {"Pretrained", "Fine-tuned"}:
            raise ValueError("Unexpected exported metrics inventory")
        temporary_zip = exports / ("." + name + ".zip.tmp")
        try:
            with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for file in sorted(bundle.rglob('*')):
                    if file.is_file(): archive.write(file, str(file.relative_to(bundle.parent)))
            with zipfile.ZipFile(temporary_zip) as archive:
                if archive.testzip() is not None: raise ValueError("ZIP CRC check failed")
                for relative, record in fingerprints.items():
                    if hashlib.sha256(archive.read(name + "/" + relative)).hexdigest() != record["sha256"]:
                        raise ValueError("ZIP hash mismatch: " + relative)
                count = len(archive.infolist())
            temporary_zip.rename(target)
        except Exception:
            temporary_zip.unlink(missing_ok=True)
            raise
        target.with_suffix('.zip.sha256').write_text(sha(target) + "  " + target.name + "\n")
        print(json.dumps(dict(archive=str(target), bytes=target.stat().st_size,
            mib=round(target.stat().st_size / 1048576, 2), files=count, verified=True,
            entry_point=name + "/index.html"), indent=2))


if __name__ == "__main__":
    main()
