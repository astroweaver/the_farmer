# The Farmer

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Docs](https://readthedocs.org/projects/the-farmer/badge/?version=latest)](https://the-farmer.readthedocs.io/en/latest/)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2310.07757-red)](https://arxiv.org/abs/2310.07757)

**Model photometry for deep, multi-wavelength galaxy surveys.**

The Farmer measures precise multi-band fluxes and morphologies by fitting parametric surface brightness profiles — point sources, exponential disks, de Vaucouleurs bulges, and composite models — simultaneously for groups of blended sources. The fitting engine is [The Tractor](https://github.com/dstndstn/tractor) (Lang et al. 2016). Given a high-resolution detection image, The Farmer recovers accurate photometry even in the most crowded fields.

The code underpins several major survey catalogs:

- **COSMOS2020** — [Weaver et al. 2022, ApJS 258, 11](https://ui.adsabs.harvard.edu/abs/2022ApJS..258...11W/abstract)
- **SHELA** — [Leung et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230100908L/abstract)
- **H20** — [Zalesky et al. 2025](https://arxiv.org/abs/2408.05296)

If you use The Farmer in published work, please cite **[Weaver et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv231007757W/abstract)**. A copy of the paper is at [`docs/The_Farmer_Weaver23.pdf`](docs/The_Farmer_Weaver23.pdf).

---

## How it works

The Farmer divides a survey mosaic into a grid of rectangular **bricks**. Within each brick, sources detected in the high-resolution image are grouped into small clusters (typically 1–5 neighbors) and fitted simultaneously. Fitted morphologies are then held fixed while fluxes are measured across every photometric band.

```
Survey Mosaic
    ↓
Brick Grid (N × M tiles)
    ↓  per brick:
Source Detection (SEP)  →  Segmentation map
    ↓
Source Grouping (morphological dilation)
    ↓  per group:
Model Determination (PointSource → SimpleGalaxy → Exp → deV → Composite)
    ↓
Forced Photometry (frozen morphology, free fluxes per band)
    ↓
HDF5 brick  +  FITS catalog
```

---

## Requirements

- Python ≥ 3.9
- [The Tractor](https://github.com/dstndstn/tractor) and [astrometry.net](https://github.com/dstndstn/astrometry.net) (install from source — see below)
- `astropy ≥ 5.0`, `h5py ≥ 3.9`, `sep == 1.2.1`, `pathos == 0.3.0`, `reproject == 0.11.0`, `regions == 0.7`, `tqdm == 4.65.0`, `matplotlib ≥ 3.7`

---

## Installation

```bash
# 1. Clone
git clone https://github.com/astroweaver/farmer.git
cd farmer

# 2. Install Tractor and astrometry.net from source (not on PyPI)
pip install git+https://github.com/dstndstn/astrometry.net
pip install git+https://github.com/dstndstn/tractor.git

# 3. Install The Farmer and remaining dependencies
pip install -e .
```

> **macOS:** run `xcode-select --install` before step 2 to ensure you have a C compiler.  
> **Linux:** ensure `gcc` and `python3-dev` (or equivalent) are installed.

---

## Getting started

### 1. Set up your data directories

```bash
mkdir -p data/{external,interim/{bricks,psfmodels,logs},output/{figures,catalogs,ancillary}}
```

### 2. Configure

Copy `config/config.py` and edit it to point to your images:

```python
PATH_DATA = '/path/to/your/data/'

DETECTION = {
    'science': '/path/to/data/external/detection_chimean.fits',
    'subtract_background': True,
    'backtype': 'flat',
    'backregion': 'mosaic',
}

BANDS = {}
BANDS['hsc_i'] = {
    'science':  '/path/to/data/external/hsc_i.fits',
    'weight':   '/path/to/data/external/hsc_i_weight.fits',
    'psfmodel': '/path/to/data/interim/psfmodels/hsc_i.fits',
    'subtract_background': True,
    'zeropoint': 31.4,
}
# add more bands ...

N_BRICKS    = (2, 4)         # 2 columns × 4 rows
MODEL_BANDS = ['hsc_i']      # bands used for morphology fitting
```

Run Python from the directory that **contains** `config/` (not from inside it).

### 3. Validate, build, detect, model, photometer

```python
import farmer

farmer.validate()              # check paths, WCS, PSF models

farmer.build_bricks()          # cut stamps from mosaics, save HDF5
farmer.detect_sources()        # detect sources
farmer.generate_models()       # fit morphologies
farmer.photometer()            # measure fluxes in all bands
```

### 4. Inspect results

```python
brick = farmer.load_brick(1)
brick.summary()

# Dig into a specific group interactively
group = farmer.quick_group(brick_id=1, group_id=42)
group.farm()   # determine models + photometry + plots
```

---

## Tips

**Start small.** Set your intended `N_BRICKS` grid in `config.py` but only build the central brick first. With `N_BRICKS = (nx, ny)`, the central brick ID is `(ny//2) * nx + (nx//2) + 1` — pass it directly to `build_bricks`:

```python
nx, ny = (2, 4)  # your N_BRICKS from config.py
central_id = (ny // 2) * nx + (nx // 2) + 1
farmer.build_bricks(brick_ids=central_id)
```

Set `NCPUS = 0` while debugging — serial mode makes log output easy to follow and catches configuration errors before committing to a full run.

**Detection image matters.** The detection image drives source positions and morphologies. Use the highest-resolution, deepest image available — a chi-mean stack of your best optical bands is ideal. The photometric bands do not need to match its pixel scale.

**PSF preparation.** The Farmer expects PSFEx-format PSF models. Use `bin/prep_psf.py` to clip, normalize, and optionally resample a raw PSF stamp. A mismatched PSF is the most common cause of large reduced chi-squared values.

**Adding bands later.** You can add photometric bands to existing bricks without rebuilding from scratch:

```python
farmer.update_bricks(bands=['irac_ch1'])
farmer.photometer(bands=['irac_ch1'])
```

**Parallel processing.** Set `NCPUS = 8` (or however many cores you have) for production runs. Groups are distributed one at a time across workers, keeping memory usage flat.

**Timeout protection.** Long-running or diverging groups can stall a run. Set `GROUP_TIMEOUT = 120` (seconds) to skip them automatically and continue.

---

## Output

| File | Location | Contents |
|---|---|---|
| `B{id}.h5` | `PATH_BRICKS/` | Full brick state: images, catalogs, models |
| `B{id}_catalog.fits` | `PATH_CATALOGS/` | Source catalog with fluxes, magnitudes, morphologies |
| `B{id}_*.png` | `PATH_FIGURES/` | Diagnostic plots (set `PLOT > 0`) |
| `B{id}_*.reg` | `PATH_ANCILLARY/` | DS9 region files |

Key catalog columns: `id`, `ra`, `dec`, `brick_id`, `group_id`, `{band}_flux`, `{band}_flux_ujy`, `{band}_mag`, `logre`, `reff`, `ellip`, `pa`, `chisq`, `rchisq`, `flag`.

---

## Documentation

Full documentation — installation, configuration reference, pipeline description, API reference, and FAQ — is in `docs/`. Build it locally with:

```bash
pip install sphinx sphinx-rtd-theme sphinx-copybutton
cd docs && make html
```

---

## Version history

- **v2.x (current, `master`)** — parallel group processing, lazy brick loading, variable PSF support, improved memory management
- **v1.x (`TheFarmer_v1`)** — code as described in Weaver et al. 2023; kept for reproducibility

---

## Citation

```bibtex
@ARTICLE{Weaver2023,
  author  = {{Weaver}, John R. and others},
  title   = "{The Farmer: A New, Forward-Modelling Photometry Pipeline}",
  journal = {arXiv e-prints},
  year    = 2023,
  eid     = {arXiv:2310.07757},
  doi     = {10.48550/arXiv.2310.07757}
}
```

Also cite The Tractor:

```bibtex
@MISC{Lang2016,
  author = {{Lang}, Dustin},
  title  = "{Tractor: Probabilistic astronomical source detection and measurement}",
  year   = 2016,
  note   = {Astrophysics Source Code Library, record ascl:1604.008}
}
```

---

## Contact

Questions, bug reports, and feature requests are welcome at [john.weaver.astro@gmail.com](mailto:john.weaver.astro@gmail.com) or via [GitHub Issues](https://github.com/astroweaver/farmer/issues).

If you plan to use The Farmer for a new survey, reaching out first is strongly recommended — we are happy to walk you through the configuration for your specific data.
