"""Tests on synthetic data (no Challenge data needed). Run: python -m pytest -q"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

import h5ad_viewer as V


def _embryo(n=6000, seed=0, genes=("Gata4", "Hand1", "Tbx5")):
    """A 3D ellipsoid point cloud, stored with rows sorted along x, as released MERFISH stages are spatially ordered."""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, 3)) * np.array([300.0, 200.0, 120.0]) + np.array([1000.0, -500.0, 40.0])
    C = C[np.argsort(C[:, 0])]
    X = rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(X=X, obs=pd.DataFrame({"celltype": rng.choice(["a", "b", "c"], n)}, index=[str(i) for i in range(n)]))
    a.var_names = pd.Index(list(genes), dtype=object)
    a.obsm["spatial_3D"] = C.astype(np.float32)
    return a


def test_random_sample_passes():
    src = _embryo()
    sub = src[V.sample_indices(src.n_obs, 2000, 1)].copy()
    r = V.check_submission(sub, src)
    assert r["ok"] and not r["coverage"]["partial_region"]
    assert r["coverage"]["ratio"] > 0.9
    assert r["notes"] == ["Consistent with a representative sample of the source stage."]


def test_contiguous_block_is_flagged():
    src = _embryo()
    r = V.check_submission(src[:2000].copy(), src)
    assert r["coverage"]["partial_region"]
    assert r["coverage"]["ratio"] < V.COVERAGE_RATIO_WARN


def test_rescale_is_flagged_but_coverage_uses_alignment():
    src = _embryo()
    sub = src[V.sample_indices(src.n_obs, 2000, 1)].copy()
    C = np.asarray(sub.obsm["spatial_3D"], dtype=np.float64)
    sub.obsm["spatial_3D"] = ((C - C.mean(0)) * 1.3 + C.mean(0)).astype(np.float32)
    r = V.check_submission(sub, src)
    assert r["frame"]["scale_mismatch"] and not r["frame"]["origin_mismatch"]
    assert r["coverage"]["aligned"] and not r["coverage"]["partial_region"]


def test_origin_shift_is_flagged():
    src = _embryo()
    sub = src[V.sample_indices(src.n_obs, 2000, 1)].copy()
    sub.obsm["spatial_3D"] = sub.obsm["spatial_3D"] + np.float32(500.0)
    assert V.check_submission(sub, src)["frame"]["origin_mismatch"]


def test_file_without_coordinates_is_explained_not_crashed():
    src = _embryo()
    t1 = ad.AnnData(X=src.X[:100].copy())
    assert V.choose_embedding(t1) == (None, 0)
    r = V.check_submission(t1, src)
    assert not r["ok"] and "no coordinates" in r["message"]


def test_embedding_preference_and_one_column_obsm_ignored():
    a = _embryo(200)
    a.obsm["X_umap"] = np.zeros((200, 2))
    a.obsm["bad"] = np.zeros((200, 1))
    assert V.usable_embeddings(a)[0] == "spatial_3D"
    assert "bad" not in V.usable_embeddings(a)


def test_gene_colouring_is_case_insensitive_and_reports_missing():
    a = _embryo(300)
    frame, legend = V.build_plot_frame(a, "spatial_3D", np.arange(50), "gene: gata4")
    assert legend == "Expression: Gata4" and frame["color"].shape == (50,)
    with pytest.raises(ValueError):
        V.build_plot_frame(a, "spatial_3D", np.arange(50), "gene: Nkx2-5")


def test_missing_folder_lists_nothing(tmp_path):
    assert V.list_h5ad_files(tmp_path / "nope") == []


def test_backed_file_round_trip(tmp_path):
    a = _embryo(500)
    path = tmp_path / "stage.h5ad"
    a.write_h5ad(path)
    backed = V._open_backed(path)
    key, dims = V.choose_embedding(backed)
    frame, _ = V.build_plot_frame(backed, key, V.sample_indices(backed.n_obs, 100, 0), "celltype")
    assert (key, dims) == ("spatial_3D", 3) and frame.shape == (100, 4)
