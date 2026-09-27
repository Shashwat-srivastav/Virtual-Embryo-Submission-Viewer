# Virtual Embryo Submission Viewer

An interactive viewer for the Virtual Embryo Challenge `.h5ad` files, with one check no
format validator makes: **does your prediction cover the embryo the way a representative
sample does, or only part of it?**

Format validators confirm gene order, cell counts and finite values. They pass a file whose
cells come from one spatial region of the embryo, because it is a perfectly valid file. The
spatial metrics then score it as a partial embryo. This tool shows that region in 3D and
measures it, locally and in seconds, before you spend an upload.

## Why this check exists

Released MERFISH stages store cells in a spatially ordered way, so taking the first N rows,
or any contiguous block of rows, takes a slab of the embryo. We did exactly that on our own
T2 heart-extrapolation submission. The same stage, the same expression and the same
coordinates, drawn at random instead of as a row block, scored:

| File | Coverage (this tool) | Real leaderboard score |
|---|---|---|
| contiguous row block of E9.5 | 55 % of a random sample's | 38.6 |
| random sample of E9.5 | 100 % | 48.94 |

That is +10.3 points on one board from which rows were submitted. The coverage check
separates the two files without any target data.

It also caught the same defect in a file we had already scored and thought was fine. Our
T2 embryo-interpolation submission (55.0) measured **61 %** coverage. The builder loaded only
the first 8,000 rows of each stage, so the cells came from one region of E7.25, and the
cell-type means used 8,000 of the later stage's 31,671 cells. Rebuilt with the same method
on all rows and a random sample:

| File | Coverage (this tool) | Real leaderboard score |
|---|---|---|
| same method, first 8,000 rows of each stage | 61 % | 55.0 |
| same method, all rows, random sample | 100 % | 57.61 |

The gain was smaller than on heart-extrap (+2.6 against +10.3), so treat coverage as a
reason to look, not as a predicted number of points.

## Two modes

**Explore a file.** Plot a released stage or your own prediction in 2D or 3D, coloured by a
metadata column or by one gene's expression (case-insensitive name search, 99th-percentile
colour clipping so sparse genes stay readable). Files open in backed mode and only the
plotted cells are read, so the larger stages stay usable on a laptop. Files without
coordinates, such as T1 predictions, open with an explanation instead of an error.

**Check a submission.** Pick your prediction and the released stage it was built from.
You get:

- **Coverage vs a random sample.** The share of the source's occupied voxels (16³ grid over
  the source's extent) that your cells reach, divided by what a random subsample of the same
  size reaches. A representative sample lands near 100 %. Below 80 % is flagged as a partial
  region.
- **RMS radius ratio and centroid offset.** Coordinates differ in scale and origin between
  embryos. A rescale or shift of more than about 15 % or 0.25 radii is flagged, and
  coverage is then computed after aligning onto the source, so the two problems don't mask
  each other.
- **A 3D overlay** of your prediction on the source stage.

## Install and run

```bash
pip install -r requirements.txt

# the app
streamlit run h5ad_viewer.py -- --data-dir /path/to/data/raw --submission-dir /path/to/your/predictions

# the check, headless
python h5ad_viewer.py --compare my_prediction.h5ad /path/to/data/raw/E9.5.h5ad

# smoke test: open every released file and build a small plot frame
python h5ad_viewer.py --check --data-dir /path/to/data/raw
```

Both folders can also be set in the sidebar, and files can be uploaded directly
(`.streamlit/config.toml` raises Streamlit's upload limit to 2 GB, because released stages
and T1 predictions exceed the 200 MB default). Tested with Python 3.12, anndata 0.12,
streamlit 1.45 and plotly 6.1.

Headless output for the block file above:

```
coverage 0.524 vs random 0.961 -> ratio 0.55 | radius ratio 0.93 | centroid offset 0.37 radii
- Covers 55% of what a random sample of the same size covers: the prediction looks like a spatial region, not the whole embryo. ...
- Centroid is 0.37 source radii from the source's centroid. Files differ in origin; ...
```

## What it does not do

- It does not score anything or predict a leaderboard number. Coverage is a diagnostic: a
  full-coverage file can still score badly, and a deliberately regional prediction is
  possible. It tells you that your file is regional, not whether that is what you wanted.
- It needs `obsm["spatial_3D"]` in both files. T1 predictions have no coordinates, so there
  is nothing spatial to check.
- It is not a format validator. Use one of those as well (several are listed on the
  Challenge's community page).
- No Challenge data is included or redistributed.

## Tests

```bash
python -m pytest -q
```

Nine tests on synthetic, spatially ordered point clouds: a random sample passes, a
contiguous block is flagged, rescale and origin shifts are flagged and aligned, files
without coordinates are explained rather than crashing, and the gene and embedding
lookups behave.

## License

MIT
