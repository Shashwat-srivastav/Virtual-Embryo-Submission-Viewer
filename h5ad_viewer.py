"""Interactive viewer for Virtual Embryo Challenge .h5ad files, with a pre-upload submission check.

Two modes:

* **Explore a file** - plot a released stage or your own prediction in 2D/3D, coloured by
  metadata or by one gene's expression. The matrix stays backed on disk; only the plotted
  cells are read.
* **Check a submission** - overlay your prediction on the stage it was built from and
  measure whether it covers that stage the way a representative sample would. A
  contiguous row block of a MERFISH file is a spatial chunk of the embryo, and the spatial
  metrics punish it; this check catches that before an upload does.

Run:
    streamlit run h5ad_viewer.py -- --data-dir /path/to/data/raw --submission-dir /path/to/predictions
    python h5ad_viewer.py --check --data-dir /path/to/data/raw          # headless smoke test
    python h5ad_viewer.py --compare pred.h5ad source.h5ad               # headless submission check
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path.cwd() / "data" / "raw"
DEFAULT_PLOT_CELLS = 5_000
MAX_PLOT_CELLS = 10_000
MAX_3D_CELLS = 5_000
PREFERRED_EMBEDDINGS = ["spatial_3D", "X_umap", "X_umap.harmony.rna", "spatial_2D", "X_pca"]

# A random subsample of the source stage defines the expected coverage; a submission
# covering less than this fraction of it is flagged as a partial region of the embryo.
COVERAGE_RATIO_WARN = 0.8
FRAME_OFFSET_WARN = 0.25   # centroid offset, in units of the source's RMS radius
FRAME_SCALE_WARN = 0.15    # |log| of the RMS-radius ratio, about 15 %


# -- files and embeddings ---------------------------------------------------------

def list_h5ad_files(directory: str | Path) -> list[Path]:
    """Return .h5ad files in a directory, sorted for stable UI behaviour; [] if it does not exist."""
    path = Path(directory).expanduser()
    if not path.is_dir():
        return []
    return sorted(path.glob("*.h5ad"), key=lambda p: p.name.lower())


def usable_embeddings(adata: Any) -> list[str]:
    """obsm keys that can be plotted (2D matrices with at least two columns), preferred first."""
    keys = []
    for key in adata.obsm.keys():
        shape = getattr(adata.obsm[key], "shape", ())
        if len(shape) == 2 and shape[1] >= 2:
            keys.append(key)
    return sorted(keys, key=lambda k: (PREFERRED_EMBEDDINGS.index(k) if k in PREFERRED_EMBEDDINGS else len(PREFERRED_EMBEDDINGS), k))


def choose_embedding(adata: Any, requested: str | None = None) -> tuple[str | None, int]:
    """Pick a plottable embedding. Returns (None, 0) when the file has none (e.g. a T1 prediction)."""
    keys = usable_embeddings(adata)
    if requested in keys:
        key = requested
    elif keys:
        key = keys[0]
    else:
        return None, 0
    return key, int(adata.obsm[key].shape[1])


def sample_indices(n_obs: int, max_cells: int, seed: int) -> np.ndarray:
    """Sample rows without replacement, sorted, deterministic for a seed."""
    if n_obs <= max_cells:
        return np.arange(n_obs)
    return np.sort(np.random.default_rng(seed).choice(n_obs, max_cells, replace=False))


def _categorical_or_numeric(values: pd.Series) -> tuple[np.ndarray, str]:
    """Prepare obs/gene values for Plotly and identify their colour mode."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95 and values.nunique(dropna=True) > 10:
        return numeric.to_numpy(dtype=float), "numeric"
    return values.astype("string").fillna("<missing>").to_numpy(), "categorical"


def build_plot_frame(adata: Any, embedding_key: str, indices: np.ndarray, color_by: str,
                     layer: str | None = None) -> tuple[pd.DataFrame, str]:
    """Materialise only the selected rows and return a Plotly-ready frame."""
    coords = np.asarray(adata.obsm[embedding_key])[indices]
    frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if coords.shape[1] >= 3:
        frame["z"] = coords[:, 2]

    if color_by.startswith("gene: "):
        gene = color_by[6:]
        names = np.asarray(adata.var_names)
        hits = np.flatnonzero(names == gene)
        if hits.size == 0:
            lower = np.flatnonzero(np.char.lower(names.astype(str)) == gene.lower())
            if lower.size == 0:
                raise ValueError(f"Gene '{gene}' is not in this file's gene panel")
            hits = lower
        source = adata.layers[layer] if layer else adata.X
        # Released files store X as CSC: select the column first, then the rows.
        values = source[:, int(hits[0])][indices]
        values = values.toarray().ravel() if hasattr(values, "toarray") else np.asarray(values).ravel()
        frame["color"] = values.astype(float)
        return frame, f"Expression: {names[hits[0]]}"

    if color_by == "None":
        frame["color"] = "All cells"
        return frame, "All cells"

    values, mode = _categorical_or_numeric(adata.obs[color_by].iloc[indices])
    frame["color"] = values
    return frame, color_by if mode == "categorical" else f"{color_by} (numeric)"


# -- submission check -------------------------------------------------------------

def rms_radius(C: np.ndarray) -> float:
    c = C - C.mean(axis=0)
    return float(np.sqrt((c ** 2).sum(axis=1).mean()))


def frame_diagnostics(sub_C: np.ndarray, ref_C: np.ndarray) -> dict:
    """Scale and origin of the submission relative to the source stage."""
    r_ref = rms_radius(ref_C)
    ratio = rms_radius(sub_C) / r_ref if r_ref > 0 else float("nan")
    offset = float(np.linalg.norm(sub_C.mean(axis=0) - ref_C.mean(axis=0)) / r_ref) if r_ref > 0 else float("nan")
    return {
        "radius_ratio": ratio,
        "centroid_offset": offset,
        "scale_mismatch": bool(abs(np.log(ratio)) > FRAME_SCALE_WARN) if ratio > 0 else True,
        "origin_mismatch": bool(offset > FRAME_OFFSET_WARN),
    }


def _occupied(C: np.ndarray, lo: np.ndarray, hi: np.ndarray, grid: int) -> set:
    cells = np.floor((C - lo) / np.where(hi > lo, hi - lo, 1.0) * grid).astype(int)
    cells = np.clip(cells, 0, grid - 1)
    return set(map(tuple, cells))


def _align(C: np.ndarray, ref_C: np.ndarray) -> np.ndarray:
    """Re-centre and isotropically rescale C onto ref_C's centroid and RMS radius."""
    r = rms_radius(C)
    scale = rms_radius(ref_C) / r if r > 0 else 1.0
    return (C - C.mean(axis=0)) * scale + ref_C.mean(axis=0)


def sample_coverage(sub_C: np.ndarray, ref_C: np.ndarray, grid: int = 16, seeds: int = 5,
                    align: bool = False) -> dict:
    """Share of the source stage's occupied voxels that the submission reaches, against a random subsample.

    The expected value is the coverage of a random subsample of the source with the same number of
    cells, averaged over `seeds`. A ratio well below 1 means the submission sits in part of the
    embryo only, which is what a contiguous row block of a spatially ordered file does.
    """
    sub_C = np.asarray(sub_C, dtype=np.float64)[:, :3]
    ref_C = np.asarray(ref_C, dtype=np.float64)[:, :3]
    if align:
        sub_C = _align(sub_C, ref_C)
    lo, hi = ref_C.min(axis=0), ref_C.max(axis=0)
    ref_vox = _occupied(ref_C, lo, hi, grid)
    inside = np.all((sub_C >= lo) & (sub_C <= hi), axis=1)
    coverage = len(_occupied(sub_C[inside], lo, hi, grid) & ref_vox) / len(ref_vox) if inside.any() else 0.0
    n = min(len(sub_C), len(ref_C))
    expected = float(np.mean([
        len(_occupied(ref_C[sample_indices(len(ref_C), n, s)], lo, hi, grid)) / len(ref_vox)
        for s in range(seeds)
    ]))
    ratio = coverage / expected if expected > 0 else float("nan")
    return {
        "coverage": coverage,
        "expected_random": expected,
        "ratio": ratio,
        "outside_source_extent": float(1.0 - inside.mean()),
        "partial_region": bool(ratio < COVERAGE_RATIO_WARN),
        "aligned": align,
    }


def check_submission(sub: Any, ref: Any, key: str = "spatial_3D") -> dict:
    """Full pre-upload check of a prediction against the stage it was built from."""
    if key not in sub.obsm or key not in ref.obsm:
        missing = [name for name, a in (("prediction", sub), ("source", ref)) if key not in a.obsm]
        return {"ok": False, "message": f"obsm['{key}'] missing in: {', '.join(missing)}. "
                "T1 predictions carry no coordinates, so there is nothing spatial to check."}
    sub_C = np.asarray(sub.obsm[key], dtype=np.float64)
    ref_C = np.asarray(ref.obsm[key], dtype=np.float64)
    frame = frame_diagnostics(sub_C, ref_C)
    aligned = frame["scale_mismatch"] or frame["origin_mismatch"]
    cov = sample_coverage(sub_C, ref_C, align=aligned)
    genes_missing = sorted(set(map(str, sub.var_names)) - set(map(str, ref.var_names)))
    notes = []
    if cov["partial_region"]:
        notes.append(f"Covers {cov['ratio']:.0%} of what a random sample of the same size covers: "
                     "the prediction looks like a spatial region, not the whole embryo. "
                     "If you subsampled rows, draw them at random instead of taking a contiguous block.")
    if frame["scale_mismatch"]:
        notes.append(f"RMS radius is {frame['radius_ratio']:.2f}x the source's. Coordinates differ in scale "
                     "between embryos; make sure a rescale is intended.")
    if frame["origin_mismatch"]:
        notes.append(f"Centroid is {frame['centroid_offset']:.2f} source radii from the source's centroid. "
                     "Files differ in origin; re-centre per file before pooling coordinates.")
    if aligned:
        notes.append("Coverage was computed after re-centring and rescaling onto the source.")
    if genes_missing:
        notes.append(f"{len(genes_missing)} prediction genes are not in the source file (first: {genes_missing[:5]}).")
    if not notes:
        notes.append("Consistent with a representative sample of the source stage.")
    return {"ok": True, "n_prediction": int(sub.n_obs), "n_source": int(ref.n_obs),
            "frame": frame, "coverage": cov, "notes": notes}


# -- app --------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive .h5ad viewer and pre-upload submission check")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="folder with the released .h5ad stages")
    parser.add_argument("--submission-dir", default=None, help="folder with your predictions (default: --data-dir)")
    parser.add_argument("--check", action="store_true", help="smoke-test every file in --data-dir without Streamlit")
    parser.add_argument("--compare", nargs=2, metavar=("PREDICTION", "SOURCE"),
                        help="run the submission check headless and print the result")
    args, _ = parser.parse_known_args(argv)
    return args


def _open_backed(path_or_buffer: Any) -> Any:
    import anndata as ad
    return ad.read_h5ad(path_or_buffer, backed="r")


def _render_figure(frame: pd.DataFrame, title: str, axis_count: int, color_by: str) -> Any:
    import plotly.express as px

    is_gene = color_by.startswith("gene: ")
    is_meta = color_by not in {"None"} and not is_gene
    kwargs: dict = {}
    if is_gene:
        # Expression is sparse and long-tailed: clip the display at the 99th percentile
        # and keep zero visibly distinct.
        expression = pd.to_numeric(frame["color"], errors="coerce").fillna(0).clip(lower=0)
        upper = max(float(np.percentile(expression, 99)) if len(expression) else 0.0, 1e-12)
        frame = frame.copy()
        frame["color"] = expression.clip(upper=upper)
        kwargs = {"color_continuous_scale": [[0.0, "#f4f4f4"], [0.02, "#fee8c8"], [0.2, "#fdbb84"],
                                             [0.5, "#e34a33"], [0.78, "#b30000"], [1.0, "#4d0000"]],
                  "range_color": [0, upper]}
    elif is_meta or color_by == "Source":
        kwargs = {"color_discrete_sequence": px.colors.qualitative.Safe}

    three_d = axis_count == 3 and "z" in frame
    plain = color_by == "None"
    if three_d:
        fig = px.scatter_3d(frame, x="x", y="y", z="z", title=title, **({} if plain else {"color": "color"}), **kwargs)
        fig.update_traces(marker={"size": 2, "opacity": 0.8, **({"color": "#2a78d6"} if plain else {})})
    else:
        fig = px.scatter(frame, x="x", y="y", title=title, render_mode="webgl",
                         **({} if plain else {"color": "color"}), **kwargs)
        fig.update_traces(marker={"size": 5, "opacity": 0.7, **({"color": "#2a78d6"} if plain else {})})
    if is_gene:
        fig.update_coloraxes(colorbar_title=f"Expression: {color_by[6:]}")
    elif not plain:
        fig.update_layout(legend_title_text="Source" if color_by == "Source" else color_by)
    fig.update_layout(height=700)
    return fig


def _pick_file(st: Any, label: str, folder: str, key: str) -> tuple[Any, str] | None:
    """Pick a file from a folder or an upload. Returns (path_or_buffer, label) or None."""
    files = list_h5ad_files(folder)
    uploaded = st.file_uploader(f"...or upload ({label})", type=["h5ad"], key=key + "_up")
    if uploaded is not None:
        return uploaded, uploaded.name
    if not files:
        st.warning(f"No .h5ad files in `{folder}`. Set the folder above or upload a file.")
        return None
    name = st.selectbox(label, [p.name for p in files], key=key)
    return Path(folder).expanduser() / name, name


def _run_app() -> None:
    import streamlit as st

    args = _parse_args()
    st.set_page_config(page_title="Virtual Embryo h5ad viewer", layout="wide")

    @st.cache_resource(show_spinner="Opening file (backed, metadata only)...")
    def load(path_or_buffer: Any) -> Any:
        return _open_backed(path_or_buffer)

    with st.sidebar:
        mode = st.radio("Mode", ["Explore a file", "Check a submission"], key="mode")
        data_dir = st.text_input("Released data folder", value=args.data_dir, key="data_dir")
        sub_dir = st.text_input("Predictions folder", value=args.submission_dir or args.data_dir, key="sub_dir")

    if mode == "Check a submission":
        _check_page(st, load, data_dir, sub_dir)
    else:
        _explore_page(st, load, data_dir, sub_dir)


def _explore_page(st: Any, load: Any, data_dir: str, sub_dir: str) -> None:
    st.title("Explore an .h5ad file")
    with st.sidebar:
        folder = st.radio("Browse", ["Released data", "Predictions"], horizontal=True, key="browse")
        picked = _pick_file(st, "File", data_dir if folder == "Released data" else sub_dir, "explore_file")
    if picked is None:
        return
    source, file_label = picked
    data = load(source)
    st.caption(f"**{file_label}**: {data.n_obs:,} cells x {data.n_vars:,} genes")

    key, dims = choose_embedding(data)
    if key is None:
        st.info("This file has no coordinates or embedding in `obsm`, so there is nothing to plot. "
                "That is expected for a T1 prediction; its content is the expression matrix only.")
        _details(st, data)
        return

    with st.sidebar, st.form("explore_controls"):
        embedding_key = st.selectbox("Coordinates", usable_embeddings(data), key="embedding")
        dims = int(data.obsm[embedding_key].shape[1])
        axis_count = st.radio("Plot dimensions", [2, 3], index=1 if dims >= 3 else 0, horizontal=True, key="axes")
        color_mode = st.radio("Colour by", ["None", "Metadata", "Gene expression"], horizontal=True, key="color_mode")
        color_by = "None"
        if color_mode == "Metadata" and len(data.obs.columns):
            color_by = st.selectbox("Metadata column", list(data.obs.columns), key="meta_col")
        elif color_mode == "Gene expression":
            gene = st.text_input("Gene name", placeholder="e.g. Gata4", key="gene")
            color_by = f"gene: {gene.strip()}"
        layer = st.selectbox("Expression layer", [None] + list(data.layers.keys()),
                             format_func=lambda x: "X (default)" if x is None else x, key="layer")
        cap = MAX_3D_CELLS if axis_count == 3 else MAX_PLOT_CELLS
        n_max = min(cap, data.n_obs)
        max_cells = st.slider("Cells to plot", min(500, n_max), n_max, min(DEFAULT_PLOT_CELLS, n_max),
                              step=100, key="n_cells") if n_max > 500 else n_max
        seed = st.number_input("Sampling seed", min_value=0, value=0, step=1, key="seed")
        render = st.form_submit_button("Render plot", type="primary")

    if "explore_render" not in st.session_state or (st.session_state.explore_render or (None,) * 3)[2] != file_label:
        st.session_state.explore_render = None
    if render:
        if color_by == "gene: ":
            st.error("Enter a gene name before rendering.")
            return
        try:
            idx = sample_indices(data.n_obs, int(max_cells), int(seed))
            frame, legend = build_plot_frame(data, embedding_key, idx, color_by, layer)
            st.session_state.explore_render = (frame, legend, file_label, embedding_key, axis_count, len(idx), color_by)
        except (KeyError, ValueError, IndexError) as exc:
            st.error(str(exc))
            return
    if st.session_state.explore_render is None:
        st.info("Choose the plot settings and click Render plot.")
    else:
        frame, _, label, key, axes, n, color_by = st.session_state.explore_render
        fig = _render_figure(frame, f"{label} · {key} · {n:,} cells", axes, color_by)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    _details(st, data)


def _check_page(st: Any, load: Any, data_dir: str, sub_dir: str) -> None:
    st.title("Check a submission against its source stage")
    st.caption("Overlay your prediction on the released stage it was built from, before you spend an upload.")
    with st.sidebar:
        pred = _pick_file(st, "Prediction", sub_dir, "pred_file")
        src = _pick_file(st, "Source stage", data_dir, "src_file")
        n_plot = st.slider("Cells to plot per file", 500, MAX_3D_CELLS, 3000, step=500, key="check_cells")
    if pred is None or src is None:
        return
    sub, ref = load(pred[0]), load(src[0])
    result = check_submission(sub, ref)
    if not result["ok"]:
        st.info(result["message"])
        return

    cov, frame = result["coverage"], result["frame"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage vs random sample", f"{cov['ratio']:.0%}",
              help="Share of the source's occupied voxels the prediction reaches, divided by what a random "
                   "subsample of the same size reaches. Near 100% for a representative sample.")
    c2.metric("RMS radius ratio", f"{frame['radius_ratio']:.2f}x")
    c3.metric("Centroid offset", f"{frame['centroid_offset']:.2f} radii")
    for note in result["notes"]:
        (st.warning if ("looks like" in note or "differ" in note) else st.write)(note)

    sub_C = np.asarray(sub.obsm["spatial_3D"], dtype=np.float64)
    ref_C = np.asarray(ref.obsm["spatial_3D"], dtype=np.float64)
    if cov["aligned"]:
        sub_C = _align(sub_C, ref_C)
    parts = []
    for name, C in ((f"Source: {src[1]}", ref_C), (f"Prediction: {pred[1]}", sub_C)):
        idx = sample_indices(len(C), n_plot, 0)
        parts.append(pd.DataFrame({"x": C[idx, 0], "y": C[idx, 1], "z": C[idx, 2], "color": name}))
    fig = _render_figure(pd.concat(parts, ignore_index=True), "Prediction over source stage", 3, "Source")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


def _details(st: Any, data: Any) -> None:
    with st.expander("File details"):
        st.write({"shape": data.shape, "obsm": list(data.obsm.keys()), "layers": list(data.layers.keys()),
                  "obs columns": list(data.obs.columns),
                  "obs_names unique": bool(pd.Index(data.obs_names).is_unique)})
        if len(data.obs.columns):
            st.dataframe(data.obs.head(100), use_container_width=True)


def _smoke_test(data_dir: str) -> None:
    files = list_h5ad_files(data_dir)
    if not files:
        raise SystemExit(f"No .h5ad files found in {data_dir}")
    for path in files:
        data = _open_backed(path)
        key, dims = choose_embedding(data)
        if key is None:
            print(f"OK {path.name}: shape={data.shape}, no embedding (nothing to plot)")
            continue
        idx = sample_indices(data.n_obs, min(100, data.n_obs), 0)
        color = data.obs.columns[0] if len(data.obs.columns) else "None"
        frame, _ = build_plot_frame(data, key, idx, color)
        print(f"OK {path.name}: shape={data.shape}, embedding={key}[{dims}], preview={frame.shape}")


def main() -> None:
    args = _parse_args()
    if args.compare:
        r = check_submission(_open_backed(args.compare[0]), _open_backed(args.compare[1]))
        if not r["ok"]:
            print(r["message"])
            return
        c, f = r["coverage"], r["frame"]
        print(f"coverage {c['coverage']:.3f} vs random {c['expected_random']:.3f} -> ratio {c['ratio']:.2f} | "
              f"radius ratio {f['radius_ratio']:.2f} | centroid offset {f['centroid_offset']:.2f} radii")
        for note in r["notes"]:
            print("-", note)
        return
    if args.check:
        _smoke_test(args.data_dir)
        return
    _run_app()


if __name__ == "__main__":
    main()
