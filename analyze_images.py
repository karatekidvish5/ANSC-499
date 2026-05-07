"""ANSC 499 dual-channel confocal image analysis.
    PREFACE: most of the code was generated using AI aid alongside our class labs. 
Expected input naming convention:
    yeast_cyano_*.czi   combined yeast + cyanobacteria images aka fusion cells
    yeast_only_*.tif    yeast-only controls for red-background thresholding (they are stained green)
    cyano_only_*.tif    cyanobacteria-only controls for red-object detection

Example run:
    python scripts/analyze_images.py --data-dir data --results-dir results

Main outputs:
    results/yeast_cell_measurements.csv
    results/per_image_summary.csv
    results/red_positive_threshold.csv
    results/cyano_control_summary.csv
    results/cyano_red_object_measurements.csv
    results/manual_validation_template.csv
    results/figures/*.png
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import KDTree
from skimage import exposure, feature, filters, measure, morphology, segmentation

try:
    import czifile
except ImportError:  # CZI support is optional unless .czi files are present
    czifile = None

warnings.filterwarnings("ignore", category=FutureWarning)

#Define all command-line settings used by the script in theory
def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify yeast-associated cyanobacterial fluorescence in dual-channel confocal images."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--czi-green-channel", type=int, default=1)
    parser.add_argument("--czi-red-channel", type=int, default=0)
    parser.add_argument("--rgb-green-channel", type=int, default=1)
    parser.add_argument("--rgb-red-channel", type=int, default=0)
    parser.add_argument("--min-yeast-area", type=int, default=60)
    parser.add_argument("--min-red-area", type=int, default=5)
    parser.add_argument("--min-red-overlap-pixels", type=int, default=5)
    parser.add_argument("--watershed-min-distance", type=int, default=7)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--control-sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--control-quantile", type=float, default=0.99)
    parser.add_argument("--red-positive-floor", type=float, default=1.0)
    parser.add_argument("--max-plot-images", type=int, default=8)
    parser.add_argument(
        "--manual-validation-csv",
        type=Path,
        default=None,
        help="Optional filled manual count CSV. If omitted, a blank template is written to results/.",
    )
    return parser.parse_args()

#ust parsing the naming convention I had for the files
def classify_image(path):
    name = path.name.lower()
    if name.startswith("yeast_only"):
        return "yeast_only"
    if name.startswith("cyano_only"):
        return "cyano_only"
    if name.startswith("yeast_cyano") or "combined" in name:
        return "combined"
    return "unknown"


def find_images(data_dir):
    suffixes = {".tif", ".tiff", ".czi"}
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in suffixes)


def clear_old_figure_outputs(figures_dir):
    for path in figures_dir.glob("*.png"):
        path.unlink()
# Converts microscopy image arrays into a consistent channel-y-x format so the script can seperate the red and green channels that the czi images would have

def to_cyx(array):
    
    array = np.squeeze(array)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        if array.shape[0] <= 4 and array.shape[1] > 16 and array.shape[2] > 16:
            return array
        if array.shape[-1] <= 4:
            return np.moveaxis(array, -1, 0)
        return array.max(axis=0, keepdims=True)

    channel_axes = [i for i, size in enumerate(array.shape) if size <= 4]
    channel_axis = channel_axes[0] if channel_axes else 0
    array = np.moveaxis(array, channel_axis, 0)
    while array.ndim > 3:
        array = array.max(axis=1)
    return array

# Loads each CZI or TIFF image, extracts the green yeast channel and red cyanobacteria channel, and records basic image metadata pretty neat
def load_channels(path, args):
    suffix = path.suffix.lower()
    if suffix == ".czi":
        if czifile is None:
            raise ImportError("Install czifile to read .czi files: pip install czifile")
        with czifile.CziFile(str(path)) as czi:
            raw = czi.asarray()
        green_index = args.czi_green_channel
        red_index = args.czi_red_channel
    else:
        raw = tifffile.imread(str(path))
        green_index = args.rgb_green_channel
        red_index = args.rgb_red_channel

    cyx = to_cyx(raw)
    if cyx.shape[0] == 1:
        green = cyx[0]
        red = np.zeros_like(green)
    else:
        green = cyx[min(green_index, cyx.shape[0] - 1)]
        red = cyx[min(red_index, cyx.shape[0] - 1)]

    return green.astype(float), red.astype(float), {
        "raw_shape": list(np.shape(raw)),
        "shape_cyx": list(cyx.shape),
        "red_channel_index": int(red_index),
        "green_channel_index": int(green_index),
    }


def image_threshold(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if np.all(values == values.flat[0]):
        return float(values.flat[0])
    return float(filters.threshold_otsu(values))


def clean_mask(mask, min_area, hole_area):
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=hole_area)
    return morphology.closing(mask, morphology.disk(1))

#Remove small labeled objects without merging adjacent watershed regions (mentioned in paper)
def remove_small_labels(labels, min_area):
    
    cleaned = morphology.remove_small_objects(labels.astype(int), min_size=min_area)
    cleaned, _, _ = segmentation.relabel_sequential(cleaned)
    return cleaned


def segment_yeast_global(green, args):
    smooth = filters.gaussian(green, sigma=args.gaussian_sigma, preserve_range=True)
    thresh = image_threshold(smooth)
    mask = clean_mask(smooth > thresh, args.min_yeast_area, args.min_yeast_area)
    labels = measure.label(mask)
    return remove_small_labels(labels, args.min_yeast_area), thresh


def segment_yeast_watershed(green, args):
    smooth = filters.gaussian(green, sigma=args.gaussian_sigma, preserve_range=True)
    thresh = image_threshold(smooth)
    mask = clean_mask(smooth > thresh, args.min_yeast_area, args.min_yeast_area)
    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(
        distance,
        labels=mask,
        min_distance=args.watershed_min_distance,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=int)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        markers = measure.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return remove_small_labels(labels, args.min_yeast_area), thresh


def detect_red_objects(red, args):
    smooth = filters.gaussian(red, sigma=max(args.gaussian_sigma / 2, 0.1), preserve_range=True)
    positive_values = smooth[smooth > 0]
    values_for_threshold = positive_values if positive_values.size > 50 else smooth.ravel()
    thresh = image_threshold(values_for_threshold)
    mask = clean_mask(smooth > thresh, args.min_red_area, args.min_red_area)
    labels = measure.label(mask)
    return remove_small_labels(labels, args.min_red_area), thresh


def normalize_for_display(image):
    if np.max(image) <= np.min(image):
        return np.zeros_like(image, dtype=float)
    lo, hi = np.percentile(image, [0.5, 99.7])
    if hi <= lo:
        hi = np.max(image)
        lo = np.min(image)
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0, 1))


def merged_rgb(green, red):
    rgb = np.zeros((*green.shape, 3), dtype=float)
    rgb[..., 0] = normalize_for_display(red)
    rgb[..., 1] = normalize_for_display(green)
    return rgb


def estimate_red_positive_threshold(control_rows, args):
    if not control_rows:
        return args.red_positive_floor, "no yeast-only controls found; floor threshold used"
    values = np.array([row["mean_red_intensity"] for row in control_rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return args.red_positive_floor, "controls had no finite red measurements; floor threshold used"
    mean_sd = values.mean() + args.control_sigma_multiplier * values.std(ddof=0)
    quantile = np.quantile(values, args.control_quantile)
    threshold = max(float(mean_sd), float(quantile), float(args.red_positive_floor))
    note = f"max(mean + {args.control_sigma_multiplier:g} SD, {args.control_quantile:g} quantile, floor)"
    return threshold, note


def centroid_counts_by_yeast(yeast_labels, red_labels):
    counts = {int(label): 0 for label in np.unique(yeast_labels) if label != 0}
    red_centroids_xy = []
    for prop in measure.regionprops(red_labels):
        y, x = prop.centroid
        yy = int(np.clip(round(y), 0, yeast_labels.shape[0] - 1))
        xx = int(np.clip(round(x), 0, yeast_labels.shape[1] - 1))
        label = int(yeast_labels[yy, xx])
        if label:
            counts[label] = counts.get(label, 0) + 1
        red_centroids_xy.append((float(x), float(y)))
    return counts, red_centroids_xy


def nearest_red_distances(yeast_props, red_centroids_xy):
    if not red_centroids_xy:
        return {int(p.label): np.nan for p in yeast_props}
    tree = KDTree(red_centroids_xy)
    out = {}
    for prop in yeast_props:
        y, x = prop.centroid
        dist, _ = tree.query((x, y))
        out[int(prop.label)] = float(dist)
    return out


def region_mean_intensity(prop):
    value = getattr(prop, "intensity_mean", None)
    return float(value if value is not None else prop.mean_intensity)

## Measures each segmented yeast cell and records a bunch of parameters that helped answered the hypothesis I made
def measure_yeast(image_name, image_type, method, yeast_labels, red_labels, green, red, threshold, args):
    red_mask = red_labels > 0
    props = measure.regionprops(yeast_labels, intensity_image=green)
    red_counts, red_centroids_xy = centroid_counts_by_yeast(yeast_labels, red_labels)
    nearest = nearest_red_distances(props, red_centroids_xy)
    rows = []

    for prop in props:
        label = int(prop.label)
        region = yeast_labels == label
        area = int(prop.area)
        red_overlap_area = int(np.count_nonzero(red_mask & region))
        mean_red = float(red[region].mean()) if area else np.nan
        mean_green = region_mean_intensity(prop)
        red_centroids_inside = int(red_counts.get(label, 0))
        red_positive_intensity = bool(mean_red > threshold)
        red_positive_overlap = bool(red_overlap_area >= args.min_red_overlap_pixels or red_centroids_inside >= 1)

        rows.append({
            "image": image_name,
            "image_type": image_type,
            "method": method,
            "yeast_label": label,
            "area_px": area,
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "mean_green_intensity": mean_green,
            "mean_red_intensity": mean_red,
            "red_overlap_area_px": red_overlap_area,
            "red_overlap_fraction": float(red_overlap_area / area) if area else np.nan,
            "red_positive_threshold": float(threshold),
            "red_positive_by_intensity": red_positive_intensity,
            "red_positive_by_overlap_or_centroid": red_positive_overlap,
            "red_positive": bool(red_positive_intensity and red_positive_overlap),
            "red_centroids_inside_yeast": red_centroids_inside,
            "nearest_red_centroid_distance_px": nearest.get(label, np.nan),
        })
    return rows

# this function helped find the bleedthrough in red channel control cyanobacterial images
def summarize_cyano_control(path, args):
    image_type = classify_image(path)
    green, red, metadata = load_channels(path, args)
    if min(green.shape) < 32:
        print(f"Skipping tiny cyano-only placeholder: {path.name} {green.shape}")
        return [], {
            "image": path.name,
            "image_type": image_type,
            "analyzable": False,
            "note": f"Skipped tiny placeholder image with shape {green.shape}",
            "red_object_count": np.nan,
            "red_object_total_area_px": np.nan,
            "red_object_mean_area_px": np.nan,
            "red_object_median_area_px": np.nan,
            "green_channel_mean": np.nan,
            "green_channel_p99": np.nan,
            "green_channel_max": np.nan,
            "mean_green_inside_red_objects": np.nan,
            "red_object_threshold": np.nan,
        }, metadata

    red_labels, red_threshold = detect_red_objects(red, args)
    object_rows = []
    green_inside_values = []
    for prop in measure.regionprops(red_labels, intensity_image=red):
        region = red_labels == prop.label
        green_inside = green[region]
        green_inside_values.extend(green_inside.tolist())
        object_rows.append({
            "image": path.name,
            "red_object_label": int(prop.label),
            "red_object_area_px": int(prop.area),
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "mean_red_intensity": region_mean_intensity(prop),
            "mean_green_intensity_in_red_object": float(green_inside.mean()) if green_inside.size else np.nan,
        })

    areas = np.array([row["red_object_area_px"] for row in object_rows], dtype=float)
    summary = {
        "image": path.name,
        "image_type": image_type,
        "analyzable": True,
        "note": "",
        "red_object_count": int(len(object_rows)),
        "red_object_total_area_px": int(np.sum(areas)) if areas.size else 0,
        "red_object_mean_area_px": float(np.mean(areas)) if areas.size else np.nan,
        "red_object_median_area_px": float(np.median(areas)) if areas.size else np.nan,
        "green_channel_mean": float(np.mean(green)),
        "green_channel_p99": float(np.percentile(green, 99)),
        "green_channel_max": float(np.max(green)),
        "mean_green_inside_red_objects": float(np.mean(green_inside_values)) if green_inside_values else np.nan,
        "red_object_threshold": float(red_threshold),
    }
    return object_rows, summary, metadata


def summarize_cells(cells):
    if not cells:
        return {
            "yeast_count": 0,
            "red_positive_yeast_count": 0,
            "red_positive_fraction": np.nan,
            "mean_red_intensity_per_cell": np.nan,
            "mean_red_overlap_fraction_per_cell": np.nan,
            "mean_area_px": np.nan,
        }
    df = pd.DataFrame(cells)
    return {
        "yeast_count": int(len(df)),
        "red_positive_yeast_count": int(df["red_positive"].sum()),
        "red_positive_fraction": float(df["red_positive"].mean()),
        "mean_red_intensity_per_cell": float(df["mean_red_intensity"].mean()),
        "mean_red_overlap_fraction_per_cell": float(df["red_overlap_fraction"].mean()),
        "mean_area_px": float(df["area_px"].mean()),
    }


def save_image_figure(path, green, red, watershed_labels):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    panels = [
        ("A", "Merged red + green", merged_rgb(green, red)),
        ("B", "Green yeast channel", green),
        ("C", "Red cyanobacteria channel", red),
        ("D", "Watershed yeast overlay", segmentation.mark_boundaries(merged_rgb(green, red), watershed_labels, color=(1, 1, 0))),
    ]
    for ax, (letter, title, img) in zip(axes.flat, panels):
        if img.ndim == 2:
            ax.imshow(normalize_for_display(img), cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"{letter}. {title}", loc="left", fontsize=11, fontweight="bold")
        ax.set_axis_off()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_method_comparison(path, green, red, global_labels, watershed_labels):
    y_slice, x_slice = zoom_slices_for_method_difference(global_labels, watershed_labels)
    green_crop = green[y_slice, x_slice]
    red_crop = red[y_slice, x_slice]
    global_crop = global_labels[y_slice, x_slice]
    watershed_crop = watershed_labels[y_slice, x_slice]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for ax, labels, title in [
        (axes[0], global_crop, "A. Otsu/global threshold"),
        (axes[1], watershed_crop, "B. Watershed"),
    ]:
        ax.imshow(segmentation.mark_boundaries(merged_rgb(green_crop, red_crop), labels, color=(1, 1, 0)))
        ax.set_title(f"{title}\nObjects in zoom: {count_labels(labels)}", loc="left", fontsize=11, fontweight="bold")
        ax.set_axis_off()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def count_labels(labels):
    return int(np.count_nonzero(np.unique(labels)))


def zoom_slices_for_method_difference(global_labels, watershed_labels, padding=35, min_size=120):
    best_label = 0
    best_count = 0
    for label in np.unique(global_labels):
        if label == 0:
            continue
        overlap = watershed_labels[global_labels == label]
        split_count = len([x for x in np.unique(overlap) if x != 0])
        if split_count > best_count:
            best_count = split_count
            best_label = int(label)

    if best_label == 0:
        ys, xs = np.nonzero(global_labels != watershed_labels)
    else:
        ys, xs = np.nonzero(global_labels == best_label)

    if len(ys) == 0 or len(xs) == 0:
        h, w = global_labels.shape
        y0 = max((h - min_size) // 2, 0)
        x0 = max((w - min_size) // 2, 0)
        return slice(y0, min(y0 + min_size, h)), slice(x0, min(x0 + min_size, w))

    h, w = global_labels.shape
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, h)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, w)
    if y1 - y0 < min_size:
        extra = min_size - (y1 - y0)
        y0 = max(y0 - extra // 2, 0)
        y1 = min(y0 + min_size, h)
    if x1 - x0 < min_size:
        extra = min_size - (x1 - x0)
        x0 = max(x0 - extra // 2, 0)
        x1 = min(x0 + min_size, w)
    return slice(y0, y1), slice(x0, x1)


def save_paired_summary_plots(summary_df, cells_df, figures_dir):
    combined = summary_df[summary_df["image_type"] == "combined"].copy()
    if combined.empty:
        return
    combined["image_short"] = combined["image"].str.replace("yeast_cyano_", "", regex=False).str.replace(".czi", "", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    method_labels = {"otsu_global": "Otsu", "watershed": "Watershed"}
    for ax, metric, ylabel, title in [
        (axes[0], "yeast_count", "Yeast objects", "A. Yeast count"),
        (axes[1], "red_positive_fraction", "Fraction red-positive", "B. Red-positive fraction"),
    ]:
        pivot = combined.pivot_table(index="image_short", columns="method", values=metric, aggfunc="mean")
        for image_id, row in pivot.iterrows():
            xs = []
            ys = []
            for method in ["otsu_global", "watershed"]:
                if method in row.index and pd.notna(row[method]):
                    xs.append(method_labels[method])
                    ys.append(row[method])
            if len(xs) == 2:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"Image {image_id}")
        ax.set_xlabel("Segmentation method")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        if metric == "red_positive_fraction":
            ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Image", loc="best", fontsize=8)
    fig.savefig(figures_dir / "paired_method_summary.png", dpi=300)
    plt.close(fig)

    metrics = [
        ("yeast_count", "Yeast objects", "paired_yeast_count.png"),
        ("red_positive_fraction", "Fraction red-positive", "paired_red_positive_fraction.png"),
    ]
    for metric, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        pivot = combined.pivot_table(index="image_short", columns="method", values=metric, aggfunc="mean")
        for image_id, row in pivot.iterrows():
            xs = []
            ys = []
            for method in ["otsu_global", "watershed"]:
                if method in row.index and pd.notna(row[method]):
                    xs.append(method_labels[method])
                    ys.append(row[method])
            if len(xs) == 2:
                ax.plot(xs, ys, marker="o", linewidth=1)
        ax.set_xlabel("Segmentation method")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " per image", loc="left", fontweight="bold")
        if metric == "red_positive_fraction":
            ax.set_ylim(0, 1)
        fig.savefig(figures_dir / filename, dpi=300)
        plt.close(fig)


def write_manual_validation_files(summary_df, args):
    combined = summary_df[summary_df["image_type"] == "combined"].copy()
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for image in sorted(combined["image"].unique()):
        image_rows = combined[combined["image"] == image].set_index("method")
        rows.append({
            "Image": image,
            "Manual yeast count": "",
            "Otsu yeast count": int(image_rows.loc["otsu_global", "yeast_count"]) if "otsu_global" in image_rows.index else "",
            "Watershed yeast count": int(image_rows.loc["watershed", "yeast_count"]) if "watershed" in image_rows.index else "",
            "Manual red-positive yeast count": "",
            "Otsu red-positive": int(image_rows.loc["otsu_global", "red_positive_yeast_count"]) if "otsu_global" in image_rows.index else "",
            "Watershed red-positive": int(image_rows.loc["watershed", "red_positive_yeast_count"]) if "watershed" in image_rows.index else "",
        })
    template = pd.DataFrame(rows)
    template_path = args.results_dir / "manual_validation_template.csv"
    template.to_csv(template_path, index=False)

    manual_path = args.manual_validation_csv
    if manual_path is None:
        default_manual = args.results_dir / "manual_validation_counts.csv"
        manual_path = default_manual if default_manual.exists() else template_path
    if not manual_path.exists():
        return template, pd.DataFrame()

    manual = pd.read_csv(manual_path)
    for col in template.columns:
        if col not in manual.columns:
            manual[col] = template[col]
    manual = manual[template.columns].copy()

    comparison = manual.copy()
    for col in ["Manual yeast count", "Manual red-positive yeast count"]:
        comparison[col] = pd.to_numeric(comparison[col], errors="coerce")
    for col in ["Otsu yeast count", "Watershed yeast count", "Otsu red-positive", "Watershed red-positive"]:
        comparison[col] = pd.to_numeric(comparison[col], errors="coerce")

    comparison["Otsu yeast absolute error"] = (comparison["Otsu yeast count"] - comparison["Manual yeast count"]).abs()
    comparison["Watershed yeast absolute error"] = (comparison["Watershed yeast count"] - comparison["Manual yeast count"]).abs()
    comparison["Otsu red-positive absolute error"] = (
        comparison["Otsu red-positive"] - comparison["Manual red-positive yeast count"]
    ).abs()
    comparison["Watershed red-positive absolute error"] = (
        comparison["Watershed red-positive"] - comparison["Manual red-positive yeast count"]
    ).abs()
    comparison["Watershed lower yeast-count error"] = (
        comparison["Watershed yeast absolute error"] < comparison["Otsu yeast absolute error"]
    )
    comparison["Watershed lower red-positive error"] = (
        comparison["Watershed red-positive absolute error"] < comparison["Otsu red-positive absolute error"]
    )
    comparison.to_csv(args.results_dir / "manual_validation_comparison.csv", index=False)

    valid = comparison.dropna(
        subset=[
            "Otsu yeast absolute error",
            "Watershed yeast absolute error",
            "Otsu red-positive absolute error",
            "Watershed red-positive absolute error",
        ]
    )
    if not valid.empty:
        validation_summary = pd.DataFrame([{
            "manual_images_evaluated": int(len(valid)),
            "mean_otsu_yeast_absolute_error": float(valid["Otsu yeast absolute error"].mean()),
            "mean_watershed_yeast_absolute_error": float(valid["Watershed yeast absolute error"].mean()),
            "mean_otsu_red_positive_absolute_error": float(valid["Otsu red-positive absolute error"].mean()),
            "mean_watershed_red_positive_absolute_error": float(valid["Watershed red-positive absolute error"].mean()),
            "watershed_lower_mean_yeast_error": bool(
                valid["Watershed yeast absolute error"].mean() < valid["Otsu yeast absolute error"].mean()
            ),
            "watershed_lower_mean_red_positive_error": bool(
                valid["Watershed red-positive absolute error"].mean() < valid["Otsu red-positive absolute error"].mean()
            ),
        }])
    else:
        validation_summary = pd.DataFrame([{
            "manual_images_evaluated": 0,
            "note": "Manual counts have not been filled in yet.",
        }])
    validation_summary.to_csv(args.results_dir / "manual_validation_summary.csv", index=False)
    return template, comparison


def process_one_image(path, args, threshold=None, make_figures=False):
    image_type = classify_image(path)
    green, red, metadata = load_channels(path, args)
    if min(green.shape) < 32:
        print(f"Skipping tiny image placeholder: {path.name} {green.shape}")
        return [], [], metadata

    red_labels, red_object_threshold = detect_red_objects(red, args)
    global_labels, global_green_threshold = segment_yeast_global(green, args)
    watershed_labels, watershed_green_threshold = segment_yeast_watershed(green, args)
    threshold = args.red_positive_floor if threshold is None else threshold

    all_cells = []
    summaries = []
    for method, labels, green_threshold in [
        ("otsu_global", global_labels, global_green_threshold),
        ("watershed", watershed_labels, watershed_green_threshold),
    ]:
        cells = measure_yeast(path.name, image_type, method, labels, red_labels, green, red, threshold, args)
        all_cells.extend(cells)
        summaries.append({
            "image": path.name,
            "image_type": image_type,
            "method": method,
            "green_threshold": float(green_threshold),
            "red_object_threshold": float(red_object_threshold),
            **summarize_cells(cells),
        })

    if make_figures:
        figures_dir = args.results_dir / "figures"
        safe_name = path.stem.replace(" ", "_")
        save_image_figure(figures_dir / f"{safe_name}_raw_channels_overlay.png", green, red, watershed_labels)
        save_method_comparison(figures_dir / f"{safe_name}_segmentation_comparison.png", green, red, global_labels, watershed_labels)

    return all_cells, summaries, metadata


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    clear_old_figure_outputs(figures_dir)

    images = find_images(args.data_dir)
    if not images:
        raise FileNotFoundError(f"No TIFF or CZI images found in {args.data_dir}")

    yeast_controls = [p for p in images if classify_image(p) == "yeast_only"]
    control_rows = []
    metadata = {}
    for path in yeast_controls:
        cells, _, meta = process_one_image(path, args, threshold=args.red_positive_floor, make_figures=False)
        control_rows.extend(row for row in cells if row["method"] == "watershed")
        metadata[path.name] = meta

    threshold, threshold_note = estimate_red_positive_threshold(control_rows, args)
    print(f"Red-positive intensity threshold: {threshold:.3f} ({threshold_note})")

    all_cells = []
    all_summaries = []
    cyano_object_rows = []
    cyano_summaries = []
    plot_count = 0
    for path in images:
        make_figures = classify_image(path) == "combined" and plot_count < args.max_plot_images
        cells, summaries, meta = process_one_image(path, args, threshold=threshold, make_figures=make_figures)
        if make_figures:
            plot_count += 1
        all_cells.extend(cells)
        all_summaries.extend(summaries)
        metadata[path.name] = meta

        if classify_image(path) == "cyano_only":
            object_rows, cyano_summary, cyano_meta = summarize_cyano_control(path, args)
            cyano_object_rows.extend(object_rows)
            cyano_summaries.append(cyano_summary)
            metadata[path.name] = cyano_meta

    cells_df = pd.DataFrame(all_cells)
    summary_df = pd.DataFrame(all_summaries)
    threshold_df = pd.DataFrame([{
        "red_positive_threshold": threshold,
        "threshold_note": threshold_note,
        "control_cell_count": len(control_rows),
        "control_sigma_multiplier": args.control_sigma_multiplier,
        "control_quantile": args.control_quantile,
        "red_positive_floor": args.red_positive_floor,
        "min_red_overlap_pixels": args.min_red_overlap_pixels,
    }])

    cells_df.to_csv(args.results_dir / "yeast_cell_measurements.csv", index=False)
    summary_df.to_csv(args.results_dir / "per_image_summary.csv", index=False)
    threshold_df.to_csv(args.results_dir / "red_positive_threshold.csv", index=False)
    pd.DataFrame(cyano_summaries).to_csv(args.results_dir / "cyano_control_summary.csv", index=False)
    pd.DataFrame(cyano_object_rows).to_csv(args.results_dir / "cyano_red_object_measurements.csv", index=False)
    with open(args.results_dir / "image_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    save_paired_summary_plots(summary_df, cells_df, figures_dir)
    write_manual_validation_files(summary_df, args)
    print(f"Saved results to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
