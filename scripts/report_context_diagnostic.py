#!/usr/bin/env python3
"""Create slice and interactive 3D views of a fixed-training-patch diagnostic."""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_context_learning import sha256
from fakect_two_stage_3d import build_scene, render_viewer


def report(experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    experiment = Path(experiment).resolve()
    records = {}
    for name, location in [('training', 'model/training.json'), ('results', 'evaluation/results.json')]:
        path = experiment/location
        if sha256(path) != Path(str(path)+'.sha256').read_text().strip():
            raise ValueError(f'{name} checksum changed')
        records[name] = json.loads(path.read_text())
        if not records[name].get('diagnostic') or not records[name].get('complete'):
            raise ValueError('This report requires a complete diagnostic experiment')
    training, results = records['training'], records['results']
    if (results['training_metadata_sha256'] != sha256(experiment/'model/training.json') or
            results['model_sha256'] != training['model_sha256'] or
            results['lock_fingerprint'] != training['lock_fingerprint']):
        raise ValueError('Evaluation and training provenance differ')
    record = results['records'][0]
    payload = experiment/'evaluation'/record['prediction_file']
    if sha256(payload) != record['prediction_sha256']:
        raise ValueError('Diagnostic prediction checksum changed')
    with np.load(payload, allow_pickle=False) as data:
        image = data['image']
        truth = data['mask'].astype(bool)
        valid = data['valid'].astype(bool)
        prediction = (data['probability'] >= results['settings']['threshold']) & valid
        spacing = data['spacing_ijk_mm']
        origin = data['origin_kji']
        sample_id = str(data['sample_id'])
    if sample_id != record['sample_id'] or not np.any(truth):
        raise ValueError('Diagnostic sample identity or target invalid')
    history_path = experiment/'model/history.json'
    if sha256(history_path) != training['history_sha256']:
        raise ValueError('Diagnostic history checksum changed')
    history = json.loads(history_path.read_text())
    output = experiment/'report'
    output.mkdir(exist_ok=False)
    (output/'INCOMPLETE').write_text('Building diagnostic review\n')
    # Slice through an actual foreground voxel nearest the target centroid.
    points = np.argwhere(truth)
    center = points[np.argmin(np.sum((points-points.mean(axis=0))**2, axis=1))]
    false_positive, false_negative = prediction & ~truth, truth & ~prediction
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    for axis, ax in enumerate(axes):
        index = int(center[axis])
        remaining = [x for x in range(3) if x != axis]
        sp = spacing[::-1]
        aspect = sp[remaining[0]]/sp[remaining[1]]
        ax.imshow(np.take(image, index, axis), cmap='gray', vmin=0, vmax=.3, origin='lower', aspect=aspect)
        colors = np.zeros(np.take(image, index, axis).shape+(4,), np.float32)
        for mask, color in [(truth & prediction, (0., .8, .55, .65)),
                            (false_positive, (1., .1, .5, .9)),
                            (false_negative, (.1, .6, 1., .9))]:
            colors[np.take(mask, index, axis)] = color
        ax.imshow(colors, origin='lower', aspect=aspect)
        ax.set_title(f"{'kji'[axis]} = {origin[axis]+index} · model grid")
        ax.set_xlabel(f"{'kji'[remaining[1]]} · patch voxels")
        ax.set_ylabel(f"{'kji'[remaining[0]]} · patch voxels")
    fig.savefig(output/'slices.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [r['diagnostic_metrics']['dice'] for r in history], color='#008866')
    axes[0].set(ylabel='Fixed training-patch Dice', xlabel='Epoch', ylim=(0,1))
    axes[1].plot(epochs, [r['diagnostic_loss'] for r in history], color='#005ca9')
    axes[1].set(ylabel='Fixed training-patch BCE + Dice loss', xlabel='Epoch')
    fig.savefig(output/'learning.png', dpi=130)
    plt.close(fig)
    masks = {'truth': truth, 'prediction': prediction, 'fp': false_positive, 'fn': false_negative}
    layers = [dict(key=key, label=label, color=color, opacity=opacity, visible=visible)
              for key,label,color,opacity,visible in [
                  ('truth','Reference aorta','#00cc8c',.45,True),
                  ('prediction','Predicted aorta','#ffcb42',.45,True),
                  ('fp','False positive','#ff197f',.8,False),
                  ('fn','False negative','#2299ff',.8,False)]]
    scene = build_scene(masks, spacing, sample_id=sample_id,
        modes=[dict(key='diagnostic', label='Training-patch fit', layers=layers,
                    explanation='Same fixed patch used for fitting and evaluation. Not unseen-case accuracy.')],
        metadata={'case_id': record['case_id'], 'model_grid_origin_kji': origin.tolist()},
        show_metrics=False)
    render_viewer(output/'three-dimensional', [scene], title='Aorta learning diagnostic', show_metrics=False,
        coordinate_description='Model-grid mm relative to the lower face of this patch; no surface smoothing',
        footer_note='Training patch only. Independent reference/prediction/error visibility and opacity. Anatomical extent remains provisional.')
    before = training['diagnostic_before']['metrics']['dice']
    after = record['metrics']['dice']
    markup = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Aorta context diagnostic</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#203040}}img{{width:100%;height:auto}}iframe{{width:100%;height:820px;border:1px solid #ccd}}.notice{{padding:1rem;background:#fff1ca}}code{{overflow-wrap:anywhere}}table{{border-collapse:collapse}}td,th{{padding:.6rem;border:1px solid #ccd}}</style>
<h1>Aorta context · learning diagnostic</h1>
<p class="notice">One fixed <strong>training</strong> patch. This measures whether the network can fit a known example; it does not measure unseen-phantom performance. Atlas extent is provisional.</p>
<p>Sample: <code>{html.escape(sample_id)}</code>. Physical patch: {html.escape(str([n*s for n,s in zip(truth.shape,spacing[::-1])]))} mm (k,j,i).</p>
<table><tr><th>Initial Dice</th><th>Selected checkpoint Dice</th><th>Epoch</th><th>Parameters</th></tr><tr><td>{before:.4f}</td><td>{after:.4f}</td><td>{training['best_epoch']}</td><td>{training['model_parameters']:,}</td></tr></table>
<h2>Learning</h2><img src="learning.png" alt="Training patch Dice and loss by epoch">
<h2>Attenuation and errors</h2><p><span style="color:#00885d">Green: correct foreground</span> · <span style="color:#d81768">Pink: false positive</span> · <span style="color:#177ec2">Blue: false negative</span>. Attenuation display: 0–0.3 cm⁻¹.</p><img src="slices.png" alt="Three orthogonal attenuation slices with reference and prediction errors">
<h2>Interactive surfaces</h2><p>Zoom, rotate, and toggle each surface or change its opacity. <a href="three-dimensional/index.html">Open viewer separately</a>.</p><iframe title="Interactive reference and prediction surfaces" src="three-dimensional/index.html"></iframe>
<p><a href="../model/training.json">Training provenance</a> · <a href="../evaluation/results.json">Evaluation metrics</a> · <a href="../input.ini">Frozen input</a></p></html>'''
    (output/'index.html').write_text(markup)
    (output/'provenance.json').write_text(json.dumps({
        'prediction_sha256': sha256(payload), 'training_metadata_sha256': sha256(experiment/'model/training.json'),
        'evaluation_sha256': sha256(experiment/'evaluation/results.json'),
        'report_script_sha256': sha256(__file__), 'renderer_sha256': sha256(ROOT/'src/fakect_two_stage_3d.py')}, indent=2)+'\n')
    (output/'INCOMPLETE').unlink()
    return output/'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(report(args.experiment))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
