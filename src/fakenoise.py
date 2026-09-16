#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakenoise.py
# Simple web viewer for NRRD volumes (2D slices) using Dash.
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import threading
import webbrowser
from pathlib import Path

import numpy as np
import nrrd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output


def info(msg: str) -> None:
	print(f"[INFO] {msg}")


def warn(msg: str) -> None:
	print(f"[WARN] {msg}")


def _load_nrrd(path: str) -> tuple[np.ndarray, dict]:
	data, header = nrrd.read(path)
	data = np.asarray(data)
	if data.ndim == 2:
		data = data[None, :, :]
	if data.ndim != 3:
		raise ValueError(f"Expected 2D or 3D NRRD, got shape {data.shape}")
	return data, header


def _slice_fig(slice2d: np.ndarray, title: str, vmin: float, vmax: float) -> go.Figure:
	fig = go.Figure(
		data=[
			go.Heatmap(
				z=np.flipud(slice2d),
				zmin=vmin,
				zmax=vmax,
				colorscale="gray",
				showscale=False,
				hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z}<extra></extra>",
			)
		]
	)
	fig.update_layout(
		title=title,
		margin=dict(l=2, r=2, t=30, b=2),
		xaxis=dict(showticklabels=False),
		yaxis=dict(showticklabels=False),
		plot_bgcolor="black",
		paper_bgcolor="black",
	)
	return fig


def run_viewer(nrrd_path: str, port: int = 8050, open_browser: bool = True) -> None:
	vol, _ = _load_nrrd(nrrd_path)
	nz, ny, nx = vol.shape
	vmax = float(np.max(vol)) if np.size(vol) else 1.0
	vmax = max(1.0, vmax)

	app = dash.Dash(__name__)
	app.layout = html.Div(
		style={
			"display": "grid",
			"gridTemplateColumns": "280px 1fr",
			"gap": "10px",
			"height": "100vh",
			"backgroundColor": "#0f1115",
			"color": "#e6e6e6",
			"padding": "10px",
		},
		children=[
			html.Div(
				style={
					"backgroundColor": "#151922",
					"borderRadius": "10px",
					"padding": "12px",
					"display": "flex",
					"flexDirection": "column",
					"gap": "10px",
				},
				children=[
					html.H3("NRRD Slice Viewer", style={"margin": "0 0 6px 0"}),
					html.Div("Sagittal (X)", style={"fontWeight": "600"}),
					dcc.Slider(id="x-slider", min=0, max=nx - 1, step=1, value=nx // 2),
					html.Div("Coronal (Y)", style={"fontWeight": "600"}),
					dcc.Slider(id="y-slider", min=0, max=ny - 1, step=1, value=ny // 2),
					html.Div("Axial (Z)", style={"fontWeight": "600"}),
					dcc.Slider(id="z-slider", min=0, max=nz - 1, step=1, value=nz // 2),
					html.Div(
						f"Loaded: {Path(nrrd_path).name} | shape={vol.shape}",
						style={"fontSize": "12px", "opacity": "0.8"},
					),
				],
			),
			html.Div(
				style={
					"display": "grid",
					"gridTemplateColumns": "1fr 1fr",
					"gridTemplateRows": "1fr 1fr",
					"gap": "10px",
				},
				children=[
					dcc.Graph(id="sagittal-view"),
					dcc.Graph(id="coronal-view"),
					dcc.Graph(id="axial-view"),
				],
			),
		],
	)

	@app.callback(
		Output("sagittal-view", "figure"),
		Output("coronal-view", "figure"),
		Output("axial-view", "figure"),
		Input("x-slider", "value"),
		Input("y-slider", "value"),
		Input("z-slider", "value"),
	)
	def update_views(x_idx: int, y_idx: int, z_idx: int):
		x_idx = int(np.clip(x_idx, 0, nx - 1))
		y_idx = int(np.clip(y_idx, 0, ny - 1))
		z_idx = int(np.clip(z_idx, 0, nz - 1))

		sagittal = vol[:, :, x_idx]
		coronal = vol[:, y_idx, :]
		axial = vol[z_idx, :, :]

		fig_x = _slice_fig(sagittal, f"Sagittal X={x_idx}", 0.0, vmax)
		fig_y = _slice_fig(coronal, f"Coronal Y={y_idx}", 0.0, vmax)
		fig_z = _slice_fig(axial, f"Axial Z={z_idx}", 0.0, vmax)
		return fig_x, fig_y, fig_z

	if open_browser:
		threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
	app.run(debug=False, port=port)


def _patient_id_from_path(img_path: Path, dataset_dir: Path) -> str:
	rel = img_path.relative_to(dataset_dir)
	stem = rel.as_posix().rsplit(".nrrd", 1)[0]
	return stem.replace("/", "__")


def _find_nrrd_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
	pairs: list[tuple[Path, Path]] = []
	for img_path in dataset_dir.rglob("*.nrrd"):
		if img_path.name.endswith(".seg.nrrd"):
			continue
		if "paired_datasets" in img_path.parts:
			continue
		mask_path = img_path.with_name(f"{img_path.stem}.seg.nrrd")
		if mask_path.exists():
			pairs.append((img_path, mask_path))
	return pairs


def generate_paired_dataset(dataset_dir: str, out_dir: str | None = None) -> Path:
	root = Path(dataset_dir).resolve()
	if not root.exists():
		raise FileNotFoundError(str(root))
	out_root = Path(out_dir).resolve() if out_dir else root / "paired_datasets"
	out_root.mkdir(parents=True, exist_ok=True)
	preview_written = False

	pairs = _find_nrrd_pairs(root)
	if not pairs:
		warn(f"No image/mask pairs found under {root}")
		return out_root

	manifest_path = out_root / "pairs.csv"
	with manifest_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["patient_id", "image", "mask", "x_index"])
		for img_path, mask_path in pairs:
			img_vol, _ = _load_nrrd(str(img_path))
			mask_vol, _ = _load_nrrd(str(mask_path))
			if img_vol.shape != mask_vol.shape:
				warn(f"Shape mismatch: {img_path} vs {mask_path}")
				continue
			pid = _patient_id_from_path(img_path, root)
			nx = img_vol.shape[2]
			for x in range(nx):
				writer.writerow([pid, img_path.as_posix(), mask_path.as_posix(), x])

			if not preview_written and nx > 0:
				try:
					from PIL import Image, ImageDraw, ImageFont
				except ImportError:
					warn("PIL not installed; skipping example PNG preview")
					preview_written = True
					continue

				x_mid = nx // 2
				slice_gray = img_vol[:, :, x_mid]
				slice_mask = mask_vol[:, :, x_mid]
				p1 = float(np.percentile(slice_gray, 1))
				p99 = float(np.percentile(slice_gray, 99))
				if p99 <= p1:
					p1 = float(np.min(slice_gray))
					p99 = float(np.max(slice_gray))
				if p99 <= p1:
					gray_u8 = np.zeros_like(slice_gray, dtype=np.uint8)
				else:
					scaled = (slice_gray.astype(np.float32) - p1) / (p99 - p1)
					scaled = np.clip(scaled, 0.0, 1.0)
					gray_u8 = (scaled * 255.0).astype(np.uint8)
				mask_u8 = (slice_mask > 0).astype(np.uint8) * 255

				left = Image.fromarray(gray_u8)
				right = Image.fromarray(mask_u8)
				border = 4
				label_h = 24
				total_w = left.width + right.width + border * 3
				total_h = left.height + border * 2 + label_h
				preview = Image.new("RGB", (total_w, total_h), (0, 0, 0))

				draw = ImageDraw.Draw(preview)
				red = (220, 38, 38)
				font = ImageFont.load_default()

				lx = border
				ly = border + label_h
				rx = border * 2 + left.width
				ry = border + label_h

				preview.paste(left.convert("RGB"), (lx, ly))
				preview.paste(right.convert("RGB"), (rx, ry))

				draw.rectangle([
					(lx - 1, ly - 1),
					(lx + left.width + 1, ly + left.height + 1)
				], outline=red, width=2)
				draw.rectangle([
					(rx - 1, ry - 1),
					(rx + right.width + 1, ry + right.height + 1)
				], outline=red, width=2)

				draw.text((lx, border), "GRAY", fill=red, font=font)
				draw.text((rx, border), "MASK", fill=red, font=font)
				preview_name = f"{pid}__x{x_mid:04d}__example.png"
				preview.save(out_root / preview_name)
				preview_written = True

	info(f"Paired dataset written to {out_root}")
	info(f"Manifest: {manifest_path}")
	return out_root


def _normalize_slice_gray(slice_gray: np.ndarray, p1: float, p99: float) -> np.ndarray:
	if p99 <= p1:
		return np.zeros_like(slice_gray, dtype=np.float32)
	scaled = (slice_gray.astype(np.float32) - p1) / (p99 - p1)
	return np.clip(scaled, 0.0, 1.0)


def _load_volume_cached(path: str, cache: dict) -> tuple[np.ndarray, float, float]:
	item = cache.get(path)
	if item is not None:
		return item
	vol, _ = _load_nrrd(path)
	p1 = float(np.percentile(vol, 1))
	p99 = float(np.percentile(vol, 99))
	if p99 <= p1:
		p1 = float(np.min(vol))
		p99 = float(np.max(vol))
	cache[path] = (vol, p1, p99)
	return cache[path]


def _preview_triplet(gray_u8: np.ndarray, mask_u8: np.ndarray, pred_u8: np.ndarray,
					 out_path: Path) -> None:
	try:
		from PIL import Image, ImageDraw, ImageFont
	except ImportError:
		warn("PIL not installed; skipping training preview")
		return

	left = Image.fromarray(gray_u8)
	mid = Image.fromarray(mask_u8)
	right = Image.fromarray(pred_u8)
	border = 4
	label_h = 24
	widths = left.width + mid.width + right.width
	total_w = widths + border * 4
	total_h = left.height + border * 2 + label_h
	preview = Image.new("RGB", (total_w, total_h), (0, 0, 0))

	draw = ImageDraw.Draw(preview)
	red = (220, 38, 38)
	font = ImageFont.load_default()

	x0 = border
	y0 = border + label_h
	x1 = x0 + left.width + border
	x2 = x1 + mid.width + border

	preview.paste(left.convert("RGB"), (x0, y0))
	preview.paste(mid.convert("RGB"), (x1, y0))
	preview.paste(right.convert("RGB"), (x2, y0))

	for x_start, img in [(x0, left), (x1, mid), (x2, right)]:
		draw.rectangle([
			(x_start - 1, y0 - 1),
			(x_start + img.width + 1, y0 + img.height + 1)
		], outline=red, width=2)

	draw.text((x0, border), "GRAY", fill=red, font=font)
	draw.text((x1, border), "MASK", fill=red, font=font)
	draw.text((x2, border), "PRED", fill=red, font=font)
	preview.save(out_path)


def train_fakenoise(csv_path: str, out_dir: str | None = None,
				 context_slices: int = 0, context_step: int = 1,
				 tol_step: float = 0.05) -> Path:
	try:
		import pandas as pd
		import tensorflow as tf
		from tensorflow.keras import layers, models
		import matplotlib.pyplot as plt
	except ImportError as e:
		raise ImportError("Missing training dependencies. Install: tensorflow pandas matplotlib") from e

	seed = 42
	epochs = 10
	batch_size = 8
	val_fraction = 0.1
	train_fraction = 0.8
	context_slices = max(0, int(context_slices))
	context_step = max(1, int(context_step))
	tol_step = float(tol_step)
	if tol_step <= 0 or tol_step > 1.0:
		raise ValueError("tol_step must be in (0, 1]")

	csv_path = str(Path(csv_path).resolve())
	if not Path(csv_path).exists():
		raise FileNotFoundError(csv_path)

	run_tag = f"ctx{context_slices}_step{context_step}"
	default_out = Path(csv_path).parent / f"fakenoise_train_{run_tag}"
	out_root = Path(out_dir).resolve() if out_dir else default_out
	out_root.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(csv_path)
	required = {"patient_id", "image", "mask", "x_index"}
	if not required.issubset(set(df.columns)):
		raise ValueError(f"CSV must contain columns: {sorted(required)}")

	patient_ids = df["patient_id"].dropna().unique().tolist()
	rng = np.random.default_rng(seed)
	rng.shuffle(patient_ids)
	n_train = max(1, int(len(patient_ids) * train_fraction))
	train_ids = set(patient_ids[:n_train])
	test_ids = set(patient_ids[n_train:])

	train_df = df[df["patient_id"].isin(train_ids)].sample(frac=1.0, random_state=seed)
	test_df = df[df["patient_id"].isin(test_ids)]
	if len(test_df) == 0:
		warn("No test patients after split; using 100% train")
		test_df = train_df.iloc[:0]

	val_size = max(1, int(len(train_df) * val_fraction))
	val_df = train_df.iloc[:val_size]
	train_df = train_df.iloc[val_size:]

	cache: dict[str, tuple[np.ndarray, float, float]] = {}
	mask_cache: dict[str, np.ndarray] = {}

	first_row = train_df.iloc[0]
	img_vol, p1, p99 = _load_volume_cached(str(first_row["image"]), cache)
	mask_vol, _ = _load_nrrd(str(first_row["mask"]))
	if img_vol.shape != mask_vol.shape:
		raise ValueError("First image/mask shape mismatch")
	shape_zyx = img_vol.shape
	shape_hw = (shape_zyx[0], shape_zyx[1])
	input_channels = 1 + 2 * context_slices

	def sample_generator(rows_df):
		for row in rows_df.itertuples(index=False):
			img_path = str(row.image)
			mask_path = str(row.mask)
			x_idx = int(row.x_index)
			img_vol, v1, v99 = _load_volume_cached(img_path, cache)
			mask_vol = mask_cache.get(mask_path)
			if mask_vol is None:
				mask_vol, _ = _load_nrrd(mask_path)
				mask_cache[mask_path] = mask_vol
			if img_vol.shape != mask_vol.shape:
				continue
			if img_vol.shape[:2] != shape_hw:
				continue
			if x_idx < 0 or x_idx >= img_vol.shape[2]:
				continue
			slice_gray = img_vol[:, :, x_idx]
			gray = _normalize_slice_gray(slice_gray, v1, v99)
			gray = gray[..., None]

			if context_slices == 0:
				slice_mask = mask_vol[:, :, x_idx]
				mask = (slice_mask > 0).astype(np.float32)[..., None]
				yield mask, gray
				continue

			mask_stack = []
			for dx in range(-context_slices, context_slices + 1):
				ix = x_idx + dx * context_step
				ix = int(np.clip(ix, 0, mask_vol.shape[2] - 1))
				mask_stack.append((mask_vol[:, :, ix] > 0).astype(np.float32))
			mask_in = np.stack(mask_stack, axis=-1)
			yield mask_in, gray

	output_signature = (
		tf.TensorSpec(shape=(*shape_hw, input_channels), dtype=tf.float32),
		tf.TensorSpec(shape=(*shape_hw, 1), dtype=tf.float32),
	)

	train_ds = tf.data.Dataset.from_generator(
		lambda: sample_generator(train_df), output_signature=output_signature
	).shuffle(256, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_ds = tf.data.Dataset.from_generator(
		lambda: sample_generator(val_df), output_signature=output_signature
	).batch(batch_size).prefetch(tf.data.AUTOTUNE)
	test_ds = tf.data.Dataset.from_generator(
		lambda: sample_generator(test_df), output_signature=output_signature
	).batch(batch_size).prefetch(tf.data.AUTOTUNE)

	inputs = layers.Input(shape=(*shape_hw, input_channels), name="mask")
	x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
	x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
	x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
	x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
	outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="gray")(x)
	model = models.Model(inputs, outputs)
	model.compile(
		optimizer="adam",
		loss="mae",
		metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
				 tf.keras.metrics.RootMeanSquaredError(name="rmse")],
	)

	history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
	metrics = {}
	if len(test_df) > 0:
		metrics = model.evaluate(test_ds, return_dict=True)

	model_path = out_root / f"fakenoise_model_{run_tag}.keras"
	model.save(model_path)

	preview_df = test_df if len(test_df) > 0 else val_df
	if len(preview_df) > 0:
		preview_written = 0
		for row in preview_df.itertuples(index=False):
			img_vol, v1, v99 = _load_volume_cached(str(row.image), cache)
			mask_vol = mask_cache.get(str(row.mask))
			if mask_vol is None:
				mask_vol, _ = _load_nrrd(str(row.mask))
				mask_cache[str(row.mask)] = mask_vol
			if img_vol.shape[:2] != shape_hw or mask_vol.shape[:2] != shape_hw:
				continue
			x_idx = int(row.x_index)
			if x_idx < 0 or x_idx >= img_vol.shape[2]:
				continue
			slice_gray = img_vol[:, :, x_idx]
			gray = _normalize_slice_gray(slice_gray, v1, v99)
			if context_slices == 0:
				slice_mask = mask_vol[:, :, x_idx]
				mask = (slice_mask > 0).astype(np.float32)
				pred = model.predict(mask[None, ..., None], verbose=0)[0, ..., 0]
				mask_u8 = (mask * 255.0).astype(np.uint8)
			else:
				mask_stack = []
				for dx in range(-context_slices, context_slices + 1):
					ix = x_idx + dx * context_step
					ix = int(np.clip(ix, 0, mask_vol.shape[2] - 1))
					mask_stack.append((mask_vol[:, :, ix] > 0).astype(np.float32))
				mask_in = np.stack(mask_stack, axis=-1)
				pred = model.predict(mask_in[None, ...], verbose=0)[0, ..., 0]
				mask_u8 = (mask_stack[context_slices] * 255.0).astype(np.uint8)
			gray_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
			pred_u8 = (np.clip(pred, 0.0, 1.0) * 255.0).astype(np.uint8)
			pid = str(row.patient_id)
			preview_path = out_root / f"train_preview_{run_tag}__{pid}__x{x_idx:04d}.png"
			_preview_triplet(gray_u8, mask_u8, pred_u8, preview_path)
			preview_written += 1
		if preview_written == 0:
			warn("No preview samples match training input shape; skipping preview")

	# Additional evaluation metrics (tolerance curve, bins, PSNR/SSIM)
	eval_df = test_df if len(test_df) > 0 else val_df
	eval_ds = test_ds if len(test_df) > 0 else val_ds
	if len(eval_df) > 0:
		thresholds = np.arange(0.0, 1.0 + 1e-9, tol_step, dtype=np.float32)
		within_counts = np.zeros_like(thresholds, dtype=np.int64)
		total_pixels = 0
		bin_edges = np.linspace(0.0, 1.0, 11, dtype=np.float32)
		bin_abs_sum = np.zeros(10, dtype=np.float64)
		bin_sq_sum = np.zeros(10, dtype=np.float64)
		bin_count = np.zeros(10, dtype=np.int64)
		psnr_sum = 0.0
		ssim_sum = 0.0
		slice_count = 0

		for mask_in, gray_gt in eval_ds:
			pred = model.predict(mask_in, verbose=0)
			err = np.abs(pred - gray_gt.numpy())
			batch_pixels = err.size
			total_pixels += batch_pixels
			flat_err = err.reshape(-1)
			for i, thr in enumerate(thresholds):
				within_counts[i] += int(np.sum(flat_err <= thr))

			gt_flat = gray_gt.numpy().reshape(-1)
			pred_flat = pred.reshape(-1)
			bin_idx = np.clip(np.digitize(gt_flat, bin_edges, right=False) - 1, 0, 9)
			abs_err = np.abs(pred_flat - gt_flat)
			sq_err = (pred_flat - gt_flat) ** 2
			for b in range(10):
				mask = bin_idx == b
				if np.any(mask):
					bin_abs_sum[b] += float(np.sum(abs_err[mask]))
					bin_sq_sum[b] += float(np.sum(sq_err[mask]))
					bin_count[b] += int(np.sum(mask))

			psnr_vals = tf.image.psnr(gray_gt, pred, max_val=1.0).numpy()
			ssim_vals = tf.image.ssim(gray_gt, pred, max_val=1.0).numpy()
			psnr_sum += float(np.sum(psnr_vals))
			ssim_sum += float(np.sum(ssim_vals))
			slice_count += int(psnr_vals.shape[0])

		frac_within = (within_counts / max(1, total_pixels)).astype(np.float64)
		tol_curve_df = pd.DataFrame({"threshold": thresholds, "fraction_within": frac_within})
		tol_curve_path = out_root / f"tolerance_curve_{run_tag}.csv"
		tol_curve_df.to_csv(tol_curve_path, index=False)

		fig = plt.figure(figsize=(6, 4))
		plt.plot(thresholds, frac_within, marker="o")
		plt.xlabel("Absolute error threshold")
		plt.ylabel("Fraction within threshold")
		plt.title("Tolerance curve")
		plt.tight_layout()
		fig.savefig(out_root / f"tolerance_curve_{run_tag}.png")
		plt.close(fig)

		bin_mae = np.divide(bin_abs_sum, np.maximum(1, bin_count))
		bin_rmse = np.sqrt(np.divide(bin_sq_sum, np.maximum(1, bin_count)))
		bin_df = pd.DataFrame({
			"bin_low": bin_edges[:-1],
			"bin_high": bin_edges[1:],
			"count": bin_count,
			"mae": bin_mae,
			"rmse": bin_rmse,
		})
		bin_df.to_csv(out_root / f"intensity_bins_{run_tag}.csv", index=False)

		fig = plt.figure(figsize=(6, 4))
		plt.plot(bin_df["bin_low"], bin_df["mae"], marker="o", label="MAE")
		plt.plot(bin_df["bin_low"], bin_df["rmse"], marker="o", label="RMSE")
		plt.xlabel("Ground truth intensity bin (low edge)")
		plt.ylabel("Error")
		plt.title("Error by intensity bin")
		plt.legend()
		plt.tight_layout()
		fig.savefig(out_root / f"intensity_bins_{run_tag}.png")
		plt.close(fig)

		if slice_count > 0:
			metrics["psnr"] = psnr_sum / slice_count
			metrics["ssim"] = ssim_sum / slice_count

	if metrics:
		metrics_path = out_root / f"test_metrics_{run_tag}.csv"
		pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

	fig = plt.figure(figsize=(6, 4))
	plt.plot(history.history.get("loss", []), label="train_loss")
	plt.plot(history.history.get("val_loss", []), label="val_loss")
	plt.xlabel("Epoch")
	plt.ylabel("MAE")
	plt.legend()
	plt.tight_layout()
	plot_path = out_root / f"training_loss_{run_tag}.png"
	fig.savefig(plot_path)
	plt.close(fig)

	info(f"Model saved to {model_path}")
	return out_root


def main() -> None:
	parser = argparse.ArgumentParser(description="NRRD viewer + paired dataset generator")
	parser.add_argument("--mode", choices=["viewer", "pair", "train"], default="viewer",
				help="Run viewer or paired dataset generator")
	parser.add_argument("--in", dest="in_path", help="Input .nrrd file (viewer mode)")
	parser.add_argument("--port", type=int, default=8050, help="Dash port")
	parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
	parser.add_argument("--dataset-dir", help="Root dataset directory (pair mode)")
	parser.add_argument("--out-dir", help="Output directory for paired dataset")
	parser.add_argument("--csv", dest="csv_path", help="CSV path for training (train mode)")
	parser.add_argument("--context", type=int, default=0,
				help="Number of neighbor mask slices on each side (train mode)")
	parser.add_argument("--context-step", type=int, default=1,
				help="Stride between neighbor slices (train mode)")
	parser.add_argument("--tol-step", type=float, default=0.05,
				help="Tolerance step for error curves (train mode)")
	args = parser.parse_args()

	if args.mode == "viewer":
		if not args.in_path:
			raise ValueError("--in is required in viewer mode")
		in_path = Path(args.in_path)
		if not in_path.exists():
			raise FileNotFoundError(str(in_path))
		run_viewer(str(in_path), port=args.port, open_browser=not args.no_open)
	elif args.mode == "pair":
		if not args.dataset_dir:
			raise ValueError("--dataset-dir is required in pair mode")
		generate_paired_dataset(args.dataset_dir, out_dir=args.out_dir)
	else:
		if not args.csv_path:
			raise ValueError("--csv is required in train mode")
		train_fakenoise(
			args.csv_path,
			out_dir=args.out_dir,
			context_slices=args.context,
			context_step=args.context_step,
			tol_step=args.tol_step,
		)


if __name__ == "__main__":
	main()
