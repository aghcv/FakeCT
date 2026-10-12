"""Protect workflow entry points, preserved evidence and generated-data boundaries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = {
    'generate_demo.py', 'prepare_cohort.py', 'train_study.py', 'preview_roi.py',
    'model_experiment.py', 'prepare_tissues.py', 'resolve_anatomy.py',
    'cohort_registry.py',
}
PUBLIC_DATA = {
    'data/arm.vti', 'data/carotid.stl', 'data/cube.stl', 'data/sphere.stl',
}
PUBLIC_VOLUMES = PUBLIC_DATA | {'tests/activity_grid_000_001_005.vti'}
GENERATED_SUFFIXES = {
    '.npz', '.npy', '.bin', '.raw', '.vti', '.nii', '.mhd', '.nrrd',
    '.vtk', '.vtp', '.vtu', '.keras', '.h5', '.hdf5', '.pt', '.pth',
    '.ckpt', '.safetensors',
}


class ProjectLayoutTests(unittest.TestCase):
    def git(self, *arguments, input=None):
        if shutil.which('git') is None or not (ROOT / '.git').exists():
            self.skipTest('Git source-checkout checks require Git and a checkout')
        result = subprocess.run(['git', *arguments], cwd=ROOT, input=input,
                                text=True, capture_output=True, timeout=20)
        return result

    # Purpose: Verify top-level workflow and documentation layout is unambiguous.
    # Inputs: Repository scripts, config/doc directories, and required workflow files.
    # Outputs: Directory/file sets match the supported public layout.
    def test_workflow_commands_and_guide_directories_are_unambiguous(self):
        self.assertEqual({p.name for p in (ROOT / 'scripts').glob('*.py')}, WORKFLOW_SCRIPTS)
        for directory, permitted in (
            ('configs', {'case-studies', 'development', 'examples', 'templates', 'tissues'}),
            ('docs', {'developer', 'reference'}),
        ):
            folders = {p.name for p in (ROOT / directory).iterdir()
                       if p.is_dir() and not p.name.startswith('.') and p.name != '__pycache__'}
            self.assertEqual(folders, permitted, directory)
        for relative in ('docs/developer/index.html', 'docs/developer/testing.html',
                         'configs/templates/coa-template.ini', 'requirements/core.txt',
                         'requirements/dev.txt', 'requirements/tensorflow.txt'):
            self.assertTrue((ROOT / relative).is_file(), relative)

    # Purpose: Verify case-study configurations and runtime atlas assets remain available.
    # Inputs: COA project metadata, public data inventory, and atlas references.
    # Outputs: Twelve unique cases resolve to phantom INIs and required atlas files are nonempty.
    def test_case_study_paths_and_runtime_atlas_remain_available(self):
        project = json.loads((ROOT / 'configs/case-studies/coa/project.json').read_text())
        cases = project['cases']
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({row['case_id'] for row in cases}), len(cases))
        for row in cases:
            path = ROOT / row['config']
            self.assertTrue(path.is_file(), row['config'])
            self.assertEqual(path.parent, ROOT / 'configs/case-studies/coa/phantoms')
        for relative in (
            'references/atlas/dpi/anatomy_atlas.csv',
            'references/atlas/dpi/cardiovascular_atlas_v1_1.csv',
            'references/atlas/dpi/vascular_routing_atlas_v1.csv',
            'references/atlas/dpi/xcat_anatomy_hierarchy.yml',
            'references/atlas/dpi/RAW_BIN_LAYOUT_SOURCE.json',
            'references/atlas/xcat/organ_ids.primary.txt',
            'references/atlas/xcat/organ_ids.040.txt',
        ):
            self.assertGreater((ROOT / relative).stat().st_size, 0, relative)
        self.assertEqual({str(p.relative_to(ROOT)) for p in (ROOT / 'data').iterdir()
                          if p.is_file() and p.name != 'README.md'}, PUBLIC_DATA)

    # Purpose: Verify relocated historical JSON retains its original bytes.
    # Inputs: Archive layout-migration manifest and moved JSON files.
    # Outputs: Every relocated JSON hash matches its recorded original fingerprint.
    def test_relocated_historical_json_retains_original_bytes(self):
        manifest = json.loads((ROOT / 'archive/research/layout-migration.json').read_text())
        checked = []
        for move in manifest['moves']:
            # Project JSON is operational configuration: its file references
            # were deliberately updated. Historical reports/snapshots are not.
            if move['from'].endswith('.json') and not move['from'].startswith('configs/'):
                with self.subTest(path=move['to']):
                    actual = hashlib.sha256((ROOT / move['to']).read_bytes()).hexdigest()
                    self.assertEqual(actual, move['original_sha256'])
                    checked.append(move['from'])
        self.assertTrue(any(Path(path).name == 'source-snapshot.json' for path in checked))
        self.assertIn('docs/integration/2026-09-16/xcat-results.json', checked)

    # Purpose: Verify Git ignores generated artifacts while retaining public examples.
    # Inputs: check-ignore over output/cache/checkpoint paths and public source assets.
    # Outputs: Exactly the generated artifact paths are ignored.
    def test_git_ignores_generated_artifacts_but_keeps_source_examples(self):
        ignored = (
            'outputs/demo/cases/demo01/source_act_1.bin',
            'scratch/experiment-notes.txt', 'data/private-phantom.csv',
            'checkpoints/model.keras', 'checkpoints/model.weights.h5',
            'checkpoints/model.pt', 'checkpoints/model.pth',
            'checkpoints/model.ckpt.index', 'checkpoints/model.safetensors',
            '.ruff_cache/result', '.mypy_cache/result', '.hypothesis/examples/result',
        )
        included = tuple(PUBLIC_VOLUMES) + (
            'configs/templates/coa-template.ini', 'configs/case-studies/coa/project.json',
            'references/atlas/dpi/anatomy_atlas.csv',
            'docs/developer/assets/roi-four-panels.png',
        )
        result = self.git('check-ignore', '--no-index', '--stdin',
                          input='\n'.join(ignored + included) + '\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), set(ignored))

    # Purpose: Verify tracked/unignored sources contain no generated payloads.
    # Inputs: Git tracked and untracked file inventory excluding public volume exceptions.
    # Outputs: No generated output, checkpoint, or NIfTI payload paths are present.
    def test_existing_git_sources_do_not_contain_generated_payloads(self):
        result = self.git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
        self.assertEqual(result.returncode, 0, result.stderr)
        unexpected = []
        for relative in set(result.stdout.split('\0')):
            path = ROOT / relative
            # An unstaged relocation can leave an old name in the index; only
            # existing files are candidates. This remains valid after staging.
            if not relative or not path.is_file() or relative in PUBLIC_VOLUMES:
                continue
            suffix = path.suffix.lower()
            if (relative.startswith(('outputs/', 'output/', 'scratch/'))
                    or suffix in GENERATED_SUFFIXES or relative.endswith('.nii.gz')
                    or '.ckpt.' in path.name):
                unexpected.append(relative)
        self.assertEqual(sorted(unexpected), [], 'Keep generated payloads in ignored storage')


if __name__ == '__main__':
    unittest.main()
