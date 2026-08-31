# The Farmer — Architectural and Scientific Review

Read-only review of `farmer/` (7 772 LOC), `config/config.py`, `bin/`. No source file was modified.
Method: Phase A orientation (full read of every source file), Phase B five independent reviewers
run as isolated subagents with no cross-talk, Phase C adversarial verification, Phase D synthesis.
See **Coverage** for what the verification pass did and did not reach.

---

> **Status: all 26 findings implemented** (2026-08-26). See
> [Implementation record](#implementation-record) at the end for what changed, what was
> corrected during implementation, and the three things left for the author to run.
> Findings text below is preserved as written at review time.

---

## Executive Summary

1. **A source that fails to fit is written to the catalog as `flux = 0.0`, not as null** — `Column(length=...)` zero-fills, `IGNORE_FAILURES = True` is the shipped default, and no flag distinguishes the two. (`farmer/image.py:3309`)
2. **The `pa` / `pa_err` catalog columns are 90° from the true position angle** — Tractor's `shape.theta` already *is* the PA east of north, and `get_params` adds another 90°. The `theta` column beside it is correct. (`farmer/utils.py:1461`)
3. **The background is never subtracted from the array handed to Tractor, while the sky is pinned at `ConstantSky(0)` and frozen** — the pedestal is absorbed into source fluxes. The config block named for this (`SUBTRACT_BW/BH/FW/FH`) is read nowhere. (`farmer/image.py:715`, `:751`)
4. **Detection aborts on the shipped config**: `~np.isscalar(background)` is bitwise-NOT on a bool, so it is truthy for scalars too and the shape assertion always fires. (`farmer/image.py:524`)
5. **A band with no weight map is silently deleted from all photometry** — the fabricated weight is zeros (the log says "ones"), which masks every pixel and makes `stage_images` skip the band at DEBUG level. (`farmer/brick.py:316-321`)

This is a real scientific instrument with a genuinely good decomposition — `Mosaic → Brick → Group`
is the right abstraction for tiled multi-band model photometry, the group-buffer and segmap-transfer
machinery is thoughtful, and the recent hardening around Ceres failures and group timeouts shows
someone who has run this at survey scale and been burned. The single biggest structural weakness is
that **failure is invisible in the output product**. `IGNORE_FAILURES = True`, `except Exception`
blocks that reject a group and return an empty result, a `write_catalog` that zero-fills rather than
nulls, and a `validate()` that checks almost nothing, compose into a pipeline that will hand you a
complete-looking FITS catalog after silently dropping an arbitrary fraction of it. Everything else
here is downstream of that: with no test of any kind in the repo and no provenance in any output,
there is nothing that would tell you it happened. The second weakness is `image.py` — a 3 344-line
`BaseImage` holding detection, staging, optimisation, statistics, plotting and all three output
formats, which is why the same logic exists in four drifted copies.

---

## Assumptions

| # | Inferred | How a wrong inference changes the conclusions |
|---|---|---|
| 1 | Repo root is `/Users/jweaver/Projects/Software/the_farmer` (the brief said `.../the_farmer/farmer`, which is the package dir). | None — all citations are repo-relative to the git root. |
| 2 | `.claude/worktrees/**` are stale duplicate checkouts and not review targets. | If they are live, findings would need re-checking there. They are gitignored. |
| 3 | Weight maps are **inverse variance**. Never stated in `config.py`; inferred from `image.py:512` (`1/sqrt(wgt)`), `image.py:2029` (`chi = residual * sqrt(weight)`), and `Image(invvar=weight)` at `image.py:747`. | If a user supplies a sigma or variance map, every uncertainty in the catalog is wrong by `wgt` or `wgt²`. Nothing checks. This is itself finding #16. |
| 4 | Realistic scale: mosaic ~10⁴×10⁴ px/band, `N_BRICKS=(2,4)` → brick ~5000×2500 px, ~10³–10⁴ groups/brick, segmap ~10⁶–10⁷ non-zero px. Taken from the shipped config and `docs/source/faq.rst:228` ("10′×10′ at 0.15″/px"). | Performance findings scale linearly with these; the ranking between them does not change. |
| 5 | Images are north-up / east-left with no rotation — `utils.py:149` asserts this in a comment ("Assumes images have no rotation"). | Finding #2's *sign* analysis is frame-dependent; the internal inconsistency it rests on is not. |
| 6 | `sep`'s `theta` is CCW from +x in radians (SExtractor `THETA_IMAGE` convention). `sep`'s own docs defer to the SExtractor manual and do not restate it. | Finding #2 does **not** depend on this — it is established from Tractor's `getRaDecBasis` alone. |

---

## Findings

```
#1 — A source that fails to fit is written to the catalog as flux = 0.0, indistinguishable from a measurement
Severity:    Critical
Category:    Correctness
Verdict:     CONFIRMED
Location:    farmer/image.py:3255-3257, 3308-3310
Found by:    Lead (Phase A), verified empirically

What's wrong
  write_catalog creates each new column lazily with Column(length=len(catalog), ...), which
  numpy-zero-fills. Any source that never reaches the assignment on the next line keeps 0.0
  forever. Three routes get you there: the `continue` at 3256 for a source with no statistics,
  a source absent from model_catalog entirely, and — the common one — every source in a group
  that run_group rejected. With conf.IGNORE_FAILURES = True (config.py:117) a rejected group
  returns an empty model_catalog and processing continues. Nothing in the output distinguishes
  "flux is zero" from "we never fit this".

Evidence
  farmer/image.py:3308-3310
      if name not in catalog.colnames:
          catalog.add_column(Column(length=len(catalog), name=name, dtype=dtype, unit=unit))
      catalog[name][catalog['id'] == source_id] = value
  farmer/image.py:3255-3257
      if not hasattr(source, 'statistics'):
          self.logger.warning(f'Source {source_id} was not fit. Skipping.')
          continue
  Verified: Column(length=3, dtype=float) -> array([0., 0., 0.]).
  Rejection routes that reach here silently: brick.py:489 (group cannot be created),
  brick.py:517 (segmap out of bounds), brick.py:525 (group larger than GROUP_SIZE_LIMIT = 5),
  utils.py:1990-1992 and 2035-2037 (any exception under IGNORE_FAILURES).

Impact
  Unfit sources enter the published catalog as 0.000 flux with 0.000 uncertainty. A downstream
  SED fit reads that as a hard non-detection with infinite precision, not as missing data. Because
  GROUP_SIZE_LIMIT = 5 drops every group of more than five blended sources, the affected
  population is exactly the crowded regions the Farmer exists to handle — the bias is systematic,
  not random. In COSMOS2020-scale bricks that is a non-trivial fraction of the catalog with no
  recoverable flag.

Fix
  Fill new columns with NaN, and add an explicit status column:
      col = Column(length=len(catalog), name=name, dtype=dtype, unit=unit)
      if np.issubdtype(col.dtype, np.floating): col[:] = np.nan
      catalog.add_column(col)
  plus a `fit_status` int column set at brick.absorb() time (0 = ok, 1 = group rejected,
  2 = optimiser failed, 3 = never attempted). Risk: downstream code that tests `flux == 0`
  will change behaviour — that is the point, but grep for it first. Integer columns cannot hold
  NaN; use -1 or a masked column there.
  Test: run one brick with GROUP_SIZE_LIMIT = 1 to force mass rejection, assert
  np.isnan(cat['hsc_i_flux']).sum() equals the number of rejected sources.

Effort: hours
```

```
#2 — The catalog `pa` and `pa_err` columns are 90 degrees from the true position angle
Severity:    Critical
Category:    Correctness
Verdict:     CONFIRMED
Location:    farmer/utils.py:1437-1439, 1461-1463
Found by:    Seat 1, and Lead independently (Seat 1 reached the right line by wrong reasoning — see below)

What's wrong
  Tractor's EllipseE.getRaDecBasis maps a unit vector along the major axis to
  (dRA, dDec) = (sin theta, cos theta). Position angle east of north is atan2(East, North), so
  shape.theta *is already* the PA east of north, in radians — and tractor's own property
  docstring says "Returns position angle in *radians*". get_params emits that correctly as the
  `theta` column at line 1438, and then emits `pa = 90 deg + theta_deg` at line 1461. Both cannot
  be a position angle. `pa` is the minor-axis angle.

Evidence
  farmer/utils.py:1437-1439
      theta_deg = np.rad2deg(theta)
      source[f'theta{suffix}'] = theta_deg * u.deg
      source[f'theta{suffix}_err'] = np.rad2deg(np.sqrt(variance_shape.theta)) * u.deg
  farmer/utils.py:1461-1463
      source[f'pa{suffix}'] = 90. * u.deg + theta_deg * u.deg
      # pa = 90deg + theta, so pa_err is exactly theta_err (was missing the sqrt)
      source[f'pa{suffix}_err'] = np.rad2deg(np.sqrt(variance_shape.theta)) * u.deg
  Convention established from tractor/ellipses.py (github.com/dstndstn/tractor, main):
      def getRaDecBasis(self):
          ''' Returns a transformation matrix that takes vectors in r_e
          to delta-RA, delta-Dec vectors. '''
          theta = self.theta; ct = math.cos(theta); st = math.sin(theta)
          ...
          G = r_deg * np.array([[ct / ab, st],
                                [-st / ab, ct]])
      G @ (0,1) = r_deg*(st, ct): the unscaled (major) axis, east component st, north component ct
      => PA_EofN = atan2(st, ct) = theta.  And:
      @property
      def theta(self):
          '''Returns position angle in *radians*'''
          return math.atan2(self.e2, self.e1) / 2.
  Cross-check that the *initialisation* is right (so the error is only on readout):
  EllipseE.fromRAbPhi uses angle = radians(2*(-phi)), hence theta = -phi. stage_models
  (image.py:886) passes phi = 90 - rad2deg(theta_sep), giving shape.theta = theta_sep - 90 deg,
  which is the correct east-of-north PA for a north-up/east-left frame. The input side is fine.

  Correction to Seat 1: it claimed the error is a mirror-plus-rotation "wrong for every galaxy
  except |theta| = 45 deg", derived from theta = -phi. That conflates the init and readout legs,
  which cancel. The real defect is a clean constant +90 deg offset. Same line, simpler mechanism,
  and the fix differs.

Impact
  Every published `pa` and `pa_exp`/`pa_dev` is the minor-axis angle. Any downstream analysis
  keyed on orientation — galaxy alignment, disc inclination, shear systematics cross-checks,
  matching morphology against an external catalog — is rotated by 90 deg. Silent: the values are
  in-range and plausible. Note `pa_err` is correct (it equals theta_err, as the comment says),
  which makes the column look internally consistent.

Fix
  Delete the 90 deg term: source[f'pa{suffix}'] = theta_deg * u.deg — at which point `pa`
  duplicates `theta` and one of them should go. Preferred: keep `theta` as the tractor-native
  parameter, redefine `pa` as an explicitly documented "position angle east of north, degrees,
  wrapped to [0,180)", and add the unit and convention to the column metadata.
  Risk: users with existing catalogs have been reading the broken column; bump the catalog
  version and say so in the release note.
  Test: build a synthetic ExpGalaxy with a known PA east of north, round-trip it through
  stage_models -> get_params, assert pa comes back equal to the input, not input+90.

Effort: minutes (the fix); hours (deciding the column contract and documenting it)
```

```
#3 — The background is never subtracted on the fitting path, while the sky is frozen at exactly zero
Severity:    Critical
Category:    Correctness
Verdict:     CONFIRMED (adversarially verified)
Location:    farmer/image.py:715-716, 745-752; config/config.py:94-97
Found by:    Seat 1

What's wrong
  stage_images fetches the raw `science` array and builds the tractor Image with
  sky=ConstantSky(0). stage_engine and force_models then call engine.freezeParam('images')
  (image.py:1126, 1182, 1314, 1336), which freezes the sky along with psf/wcs/photocal. So the
  fit is told as a hard constraint that the sky is exactly zero, on data from which the pipeline
  has just measured a non-zero globalback and then discarded it. `subtract_background` is honoured
  in exactly two places — detection (sep.extract(image-background), image.py:540) and plotting —
  so the diagnostic figure shows a background-subtracted image that the fit never saw.

Evidence
  farmer/image.py:715-716
              data = self.get_image(band=band, imgtype=data_imgtype)
              data[np.isnan(data)] = 0
  farmer/image.py:745-752
              self.images[band] = Image(
                  data=data,
                  invvar=weight,
                  psf=psfmodel,
                  wcs=read_wcs(self.get_wcs(band=band, imgtype=data_imgtype)),
                  photocal=FluxesPhotoCal(band),
                  sky=ConstantSky(0)
              )
  grep -rn 'SUBTRACT_' farmer/ bin/ config/  ->  only the four definitions in config.py:94-97,
  under the header "# Background Subtraction for Photometry". Read nowhere in the package.
  docs/source/configuration.rst:266-289 documents them as live parameters.
  farmer/__init__.py:549 carries the author's own `#TODO make sure background is dealt with`
  immediately before brick.process_groups.
  All four shipped bands set 'subtract_background': True (config.py:28, 38, 48, 60).

Impact
  For a flat pedestal b (counts/px) the linear flux estimator picks up an additive bias
  b * sum(w*m)/sum(w*m^2) = b * N_eff, where N_eff is the PSF effective area. Measured for an
  HSC-like PSF (0.168"/px, 0.6" FWHM): N_eff = 28.9 px. Additive and band-dependent, so it is
  negligible for bright objects and dominant for faint ones — it distorts colours precisely in
  the faint regime the Farmer targets, biasing SED shapes and photo-z. It cannot be undone from
  the catalog because per-source N_eff is not written out.
  Caveat, stated plainly: if the input mosaics were already sky-subtracted upstream (routine for
  HSC coadds and UltraVISTA stacks) then globalback ~ 0 and the realised error is small. That is
  very likely why this has gone unnoticed. It caps the observed impact; it does not make the code
  correct, and it silently depends on a property of the input that nothing checks.

Fix
  In stage_images, before constructing the Image:
      bkg = self.get_background(band) if self.get_property('subtract_background', band=band) else 0
      data = data - bkg          # NOT -=; get_image returns the live Cutout2D array and
                                 # stage_images runs once per decision-tree stage
  Or, alternatively, stop freezing the sky and let Tractor fit it per group.
  Risk: if the mosaics are already subtracted this is a no-op — confirm globalback ~ 0 first, so
  you know which regime you are in. Either way, wire up SUBTRACT_B*/F* or delete them from
  config.py and the docs.
  Test: after staging, assert np.median(images[b].data[images[b].invvar>0]) is consistent with
  zero to within globalrms/sqrt(N); re-fit one bright isolated source both ways and confirm the
  flux moves by ~b*N_eff and no more.

Effort: hours
```

```
#4 — Detection aborts on the shipped config: a bitwise-NOT on a bool makes the shape assertion always fire
Severity:    High
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/image.py:522-525
Found by:    Seat 3, and Lead independently

What's wrong
  `~np.isscalar(background)` applies Python's bitwise-NOT to a bool: ~True == -2 and ~False == -1,
  both truthy. The elif branch is therefore taken for every non-None background, including a
  scalar one, and the assertion then compares np.shape(scalar) == () against the image shape and
  fails. The shipped config gives DETECTION backtype='flat', so get_background returns the scalar
  globalback and Brick.extract passes it straight in.

Evidence
  farmer/image.py:522-525
          if background is None:
              background = 0
          elif ~np.isscalar(background):
              assert np.shape(background)==np.shape(image), f'Background {np.shape(background)} does not have the same shape as image {np.shape(image)}!'
  Verified: ~True = -2, bool(~True) = True; ~False = -1, bool(~False) = True.
  Simulated all four cases — float32 scalar, python float, matching array, mismatched array:
  the elif branch is entered in every one; the assert passes only for the matching array.
  Trigger path: config.py:70-73 DETECTION{'backtype':'flat','subtract_background':True} ->
  mosaic.py:153-157 estimate_background (backregion='mosaic') -> image.py:469
  set_property(background.globalback, 'background', band) -> brick.py:380-382
  `if self.properties[band]['subtract_background']: background = self.get_background(band)` ->
  image.py:680-681 returns the scalar -> assert fires.

Impact
  farmer.detect_sources() raises AssertionError on the shipped example configuration, i.e. step 2
  of bin/example_script.py. Any user following the documented quickstart with a flat background
  hits it. Under `python -O` the assert vanishes and a scalar background works fine, which is
  presumably why this is not universally fatal in practice — but it means behaviour depends on
  the interpreter's optimisation flag.

Fix
      elif not np.isscalar(background):
  Risk: none; this restores the intended semantics.
  Test: call _extract with a scalar background and assert it returns a catalog rather than raising.

  Scope correction (made while implementing). An earlier draft of this finding claimed the same
  idiom was also broken at image.py:629, :659, :2487 and :2605, which use `A & ~B`. That is
  wrong and has been withdrawn. `A & ~B` is truth-equivalent to `A and not B` for every bool
  pair (True & ~True = 1 & -2 = 0; True & ~False = 1 & -1 = 1), so those four guards behave
  correctly. Only the STANDALONE `~np.isscalar(...)` is broken, because bool(~True) and
  bool(~False) are both True with nothing to mask the sign bit. The four `& ~` sites were still
  rewritten to `and not` for legibility -- no behaviour change.
  One genuine bug was found alongside them: generate_mask (image.py:659) tested
  `'weight' in self.data[band]` and raised "Cannot overwrite exiting weight" when asked to
  generate a MASK, so it refused whenever a weight existed and never guarded the mask at all.
  Corrected to test `'mask'`.

Effort: minutes
```

```
#5 — A band with no weight map is fabricated as zeros and silently deleted from all photometry
Severity:    High
Category:    Bug
Verdict:     CONFIRMED (adversarially verified, with git archaeology)
Location:    farmer/brick.py:315-321
Found by:    Seat 3 and Seat 1, independently

What's wrong
  When a mosaic has no weight map, add_band fabricates one by copying the science cutout and
  multiplying by zero. The log message says "generated as ones". _condition_band_data then does
  `mask_bool = mask_bool | (weight <= 0)` (brick.py:159), marking every pixel of that band masked;
  stage_images sees np.sum(weight) == 0 and skips the band with a DEBUG-level message. The band
  vanishes from the fit with no warning.

Evidence
  farmer/brick.py:315-321
          # if weights or masks dont exist, make them as dummy arrays
          if 'weight' not in self.data[mosaic.band]:
              self.logger.debug(f'... data \"weight\" subimage generated as ones at {cutout.input_position_original}')
              cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
              cutout.data *= 0.
              self.data[mosaic.band]['weight'] = cutout
              self.headers[mosaic.band]['weight'] = subheader
  Reproduced end to end with numpy + astropy Cutout2D: weight becomes all -0.0,
  _condition_band_data yields mask_bool.mean() == 1.0 and sum(weight) == 0, and stage_images
  (image.py:741-743) skips the band at DEBUG level.
  Git shows this is a regression, not a design choice: commit 748b796 "avoids making large filler
  arrays when bricking" collapsed four distinct fallbacks into one `cutout.data *= 0.` pattern.
  Before it the weight fallback was `np.ones_like(mosaic.data['science'])` — genuinely ones. The
  refactor flipped 1 -> 0 and left the stale log line behind. The "Dummy zero-valued arrays"
  docstring is post-hoc (748b796 is an ancestor of docstring commit 908c049).
  Reachable on a default run: shipped DETECTION (config.py:66-73) has no 'weight' key.

Impact
  Any band without a weight map contributes nothing to the fit, and get_params (utils.py:1489)
  then masks its outputs to exactly 0: {band}_flux, {band}_flux_err, {band}_flux_ujy and
  {band}_flux_ujy_err are all 0.0 in the catalog. Composed with #1, the user gets a full-looking
  photometric column that is uniformly zero, announced only in a DEBUG log. For a survey where
  one band ships without a weight map, that is the whole band.

Fix
  Restore unit weights and say so:
      cutout.data[:] = 1.0
      self.logger.warning(f'{mosaic.band} has no weight map; using unit inverse-variance. '
                          f'Uncertainties in this band are NOT calibrated.')
  Better still, derive it: generate_weight (image.py:602) already builds 1/rms**2 from the clipped
  RMS and is currently dead code — call it here.
  Risk: unit invvar makes that band's chi2 meaningless in absolute terms, which affects the
  decision tree if the band is in MODEL_BANDS. Prefer generate_weight, and refuse to put a
  weightless band in MODEL_BANDS.
  Test: build a brick from a band with no weight configured; assert the band survives into
  self.images and that its catalog fluxes are non-zero.

Effort: minutes (fix) + hours (decide the weight policy)
```

```
#6 — Every band's model and chi image is taken from band index 0, because a list is compared to a string
Severity:    High
Category:    Correctness
Verdict:     CONFIRMED
Location:    farmer/image.py:1577-1583
Found by:    Seat 3 and Seat 1, independently; verified empirically by Lead

What's wrong
  self.engine.bands is a Python list (set at image.py:1125, 1181, 1313). `model_bands == band`
  compares a list to a string, which is always False, and False is then used as an integer index.
  Every call therefore returns image 0 regardless of which band the loop is on.

Evidence
  farmer/image.py:1577-1583
                      nparam = self.engine.getCatalog().numberOfParams() - np.sum(np.array(bands)!=band)
                      model_bands = self.engine.bands
                      src_model = self.engine.getModelImage(model_bands == band)
                      chi_model = self.engine.getChiImage(model_bands == band)
                      rchi2_model = np.sum(chi_model**2 * src_model) / np.sum(src_model)
                      rchi2_model_top.append(np.sum(chi_model**2 * src_model))
                      rchi2_model_bot.append(np.sum(src_model))
  Verified for the shipped MODEL_BANDS = ['hsc_i','hsc_z','uvista_ks']:
      ('hsc_i','hsc_z','uvista_ks') == 'hsc_z'  ->  False  ->  index 0  ->  hsc_i
  for all three bands. (Even as a numpy array it would give a boolean mask, not an index.)

Impact
  The group-level per-band `rchisqmodel` statistic is computed from the first staged band for every
  band, and the totals at image.py:1628 sum three copies of the same quantity. That value is
  written to the catalog (`{band}_rchisqmodel`, `total_rchisqmodel`) and printed on every
  plot_summary panel, where it reads as a per-band goodness-of-fit. It is not.
  Bounded, and important to state: the *decision tree* reads `['total']['rchisq']`
  (image.py:1381, 1415-1418, 1450-1452), which is computed from chi2/ndof and is unaffected. Model
  selection is not corrupted by this — only the reported model-weighted chi2 diagnostic.

Fix
      idx = model_bands.index(band)
      src_model = self.engine.getModelImage(idx)
      chi_model = self.engine.getChiImage(idx)
  While there, nparam at 1577 subtracts (n_bands - 1) from the whole group's parameter count;
  for N sources across B bands the per-band count should drop N*(B-1) flux parameters, not (B-1).
  Risk: none for the index fix. The nparam fix changes ndof and hence the reported rchisq — do it
  as a separate, announced change.
  Test: stage two bands with deliberately different data, assert rchisqmodel differs between them.

Effort: minutes
```

```
#7 — map_discontinuous computes the fast path, then unconditionally throws it away; force_simple is a no-op
Severity:    High
Category:    Performance
Verdict:     CONFIRMED
Location:    farmer/utils.py:778-847
Found by:    Lead (Phase A)

What's wrong
  The same-shape/same-pixel-scale branch (778-812) and the force_simple branch (814-833) each
  build `outdict`. Neither returns. Execution falls through to 839-845, which reassigns outdict
  from map_ids_to_coarse_pixels or parallel_process. Both fast paths are dead: the cheap direct
  index copy is computed and discarded, and conf.FORCE_SIMPLE_MAPPING has no effect at all.

Evidence
  farmer/utils.py:778
      if (array.shape == out_shape) & (np.abs(scl_in - scl_out).max() < 0.001):
  farmer/utils.py:814
      elif force_simple:
  farmer/utils.py:839-845
      if conf.NCPUS == 0:
          logger.info('Mapping to different resolution using single-core vectorized reprojection')
          outdict = map_ids_to_coarse_pixels(array, out_wcs, in_wcs)
      else:
          logger.info(f'Mapping to different resolution using multiprocessing (NCPU = {conf.NCPUS})')
          logger.warning('Multiprocessing may consume significant memory due to WCS object copying')
          outdict = parallel_process(array, out_wcs, in_wcs, n_processes=conf.NCPUS)
  There is no `return` and no `else` between 812 and 839.

Impact
  transfer_maps (image.py:2938-2939) calls this twice per band — once for the segmap, once for
  the groupmap. The shipped config has hsc_i and hsc_z on the *same* grid as the detection image,
  so both should take the pure-indexing fast path and instead take the full per-pixel WCS
  round-trip: for every non-zero segmap pixel, four corners transformed pixel->world->pixel, then
  a Python double loop accumulating into a set(). At ~10⁶–10⁷ non-zero pixels per brick that is
  the dominant non-Tractor cost of detection, and it is pure waste for same-grid bands.
  Also note the log line is actively misleading: it says "Mapping to different resolution" even
  when the resolutions are identical.

Fix
  Add `return outdict` at the end of each of the two fast branches (after line 812 and after 833),
  or restructure as if/elif/else. Two lines.
  Risk: very low, but the fast path has effectively never executed, so it is untested — verify
  its output equals the slow path's before trusting it (see the Fix's test).
  Test: for a same-grid band, assert map_discontinuous's dict equals the slow-path dict key for
  key, then time both. This is exactly the kind of thing the regression floor in #20 should pin.

Effort: minutes (fix) + hours (validating the previously-dead path)
```

```
#8 — get_fwhm returns the minimum of the measured FWHM and 1.0, so it always returns <= 1 pixel
Severity:    High
Category:    Correctness
Verdict:     CONFIRMED
Location:    farmer/utils.py:427-447
Found by:    Lead (Phase A), verified empirically

What's wrong
  The final line takes np.nanmin of [1.0, fwhm] where the docstring and the trailing comment both
  describe a floor. It is a ceiling. Any real PSF or source is wider than one pixel, so the
  function returns exactly 1.0 for every input it is ever given.

Evidence
  farmer/utils.py:441-447
      dx, dy = np.nonzero(img > np.nanmax(img)/2.)
      try:
          fwhm = np.mean([dx[-1] - dx[0], dy[-1] - dy[0]])
      except (IndexError, ValueError):
          # Empty array or single pixel - cannot compute FWHM
          fwhm = np.nan
      return np.nanmin([DEFAULT_FWHM_MIN, fwhm])  # Cap to prevent unrealistic values
  Measured on Gaussians: true FWHM 9.42 px -> get_fwhm 1.0; true FWHM 2.355 px -> get_fwhm 1.0.
  The docstring at 431-432 says "Returns at most DEFAULT_FWHM_MIN (1 pixel) to prevent
  unrealistically small values" — which is self-contradictory and describes neither behaviour.

Impact
  Three consumers, all of which silently produce a constant:
   - image.py:1574  nres_elem = area / get_fwhm(psf.img)**2   -> divides by 1
   - image.py:1689  nres_elem = (get_fwhm(data)/get_fwhm(psf.img))**2 -> identically 1.0
     so the `nres` column ("resolution elements") is 1.0 for every source in every band, and the
     group-level ntotalres_elem is just the band count. It is written to the catalog and printed
     on every plot_summary panel as a physical quantity.
   - image.py:2507  hwhm = get_fwhm(psf.img)/2 * pixscl -> the "beam" circle on summary plots is
     drawn at half a pixel for every band regardless of the actual PSF.
  utils.py:449 get_resolution is built on it and is dead code (referenced only by autodoc).
  Not used in model selection, so no fit is affected.

Fix
      return np.nanmax([DEFAULT_FWHM_MIN, fwhm])
  and fix the docstring to say "floor". Separately, the estimator itself is crude: np.nonzero
  returns sorted indices, so dx[-1]-dx[0] is the full row extent of all above-half pixels, not a
  profile width — it overestimates for an elongated or blended stamp. For PSF work prefer a
  proper second-moment or radial-profile FWHM.
  Risk: nres changes from a constant to a real number, so any archived catalog's nres column
  becomes non-comparable. Say so in the release note.
  Test: assert get_fwhm(gaussian(sigma=4)) is within 10% of 2.355*4.

Effort: minutes
```

```
#9 — Mosaic centre and size are computed with (x, y) and (ny, nx) transposed
Severity:    High
Category:    Correctness
Verdict:     CONFIRMED
Location:    farmer/mosaic.py:121-125
Found by:    Lead (Phase A), verified empirically

What's wrong
  wcs.array_shape is (ny, nx); wcs.pixel_to_world takes (x, y). The code passes
  (arr_shape[0]/2, arr_shape[1]/2) = (ny/2, nx/2), i.e. the axes swapped. The next line multiplies
  the (ny, nx) shape by proj_plane_pixel_scales, which returns [scale_x, scale_y] — also
  mismatched. Both are correct only for a square image with square pixels.

Evidence
  farmer/mosaic.py:121-125
              arr_shape = self.wcs.array_shape
              self.position = self.wcs.pixel_to_world(arr_shape[0]/2., arr_shape[1]/2.)
              # upper = self.wcs.pixel_to_world(arr_shape[0], arr_shape[1])
              # lower = self.wcs.pixel_to_world(0, 0)
              self.size = arr_shape * self.pixel_scale
  Measured on a 10000 x 4000 TAN mosaic at COSMOS, 0.168"/px:
      as coded : 10h01m01.6s +02d20m44.9s
      correct  : 10h00m27.9s +02d12m21.0s
      separation = 11.88 arcmin

Impact
  Two consequences. (a) validate() and every mosaic load log a wrong field centre — the first
  number a user checks when setting up a survey, and it is silently wrong for any non-square
  mosaic. (b) Mosaic.spawn_brick(brick_id=None) (mosaic.py:198-227) passes self.position and
  self.size to Brick(), so the manual position/size path cuts a brick from the wrong sky location.
  The main brick_id path uses load_brick_position instead and is unaffected, which is why this has
  not surfaced — but it means the manual entry point is broken.

Fix
      ny, nx = self.wcs.array_shape
      self.position = self.wcs.pixel_to_world(nx/2., ny/2.)
      scl = self.pixel_scale                     # [scale_x, scale_y]
      self.size = (ny * scl[1], nx * scl[0])     # (dec_height, ra_width) -- match Brick's order
  Note Brick documents size as (dec_height, ra_width) (brick.py:38-40); make Mosaic agree.
  Risk: low; the brick_id path does not consume these.
  Test: on a deliberately non-square synthetic WCS, assert mosaic.position.separation(true_centre)
  is under one pixel.

Effort: minutes
```

```
#10 — Group spawning rescans the entire brick groupmap four times per group
Severity:    High
Category:    Performance
Verdict:     CONFIRMED (adversarially verified, both parties measured independently)
Location:    farmer/group.py:88-100
Found by:    Seat 2

What's wrong
  get_image('groupmap', 'detection') returns the full brick-sized array. Lines 90 and 97 each
  materialise `groupmap == group_id` over the whole brick — four full passes (two compares, one
  sum, one nonzero) to obtain one bounding box and one pixel count. This runs once per group, and
  because process_groups builds groups in a generator consumed by pool.imap (brick.py:622), it
  executes in the *parent* process even under multiprocessing — so it is serial no matter what
  NCPUS is set to. The author's own `#TODO -- save this somewhere` is on line 90.

Evidence
  farmer/group.py:88-100
              # use groupmap from brick to get position and buffsize
              groupmap = image.get_image(imgtype='groupmap', band='detection')
              group_npix = np.sum(groupmap==group_id) #TODO -- save this somewhere
              ...
                      idx, idy = (groupmap == group_id).nonzero()
              ...
                  xlo, xhi = np.min(idx), np.max(idx)
                  ylo, yhi = np.min(idy), np.max(idy)
  Measured (this machine, int32):
      5000x2500 = 12.5 Mpx:  np.sum 3.3 ms + nonzero 24.6 ms = 28.0 ms per spawn_group
      4000x4000 = 16.0 Mpx (the size docs/source/faq.rst:228 implies): 39.4 ms per group
  Current cost per brick:  1 000 groups -> 0.5 min;  5 000 -> 2.3 min;  10 000 -> 4.7 min
      (at 16 Mpx: 10 000 groups -> 6.6 min)
  Proposed cost: one pass over the groupmap computing every group's bbox and count at once —
  np.nonzero + np.minimum.at/np.maximum.at/np.bincount — measured at 0.05 s TOTAL for 10 000
  groups. Valid because dilate_and_group renumbers groups to a contiguous 1..n (utils.py:396-398).
  Speedup on this step: ~5 600x.
  What would make this estimate wrong: if the median Tractor fit per group is long (say >2 s),
  this is only a few percent of brick wall-clock and drops to a nice-to-have. Nothing in the repo
  or docs pins the per-group fit time, so this is the open number.

Impact
  Minutes per brick of pure array scanning, unparallelisable, paid twice in the shipped flow
  (generate_models at __init__.py:558 and photometer at :617/:619). It also serialises the
  producer side of the process pool, capping the achievable speedup from NCPUS.

Fix
  Precompute once, on the brick, right after identify_groups:
      ys, xs = np.nonzero(groupmap); g = groupmap[ys, xs]
      self.group_bboxes = {gid: (ymin, ymax, xmin, xmax, npix)}   via np.minimum.at / np.bincount
  and have Group.__init__ read the dict. Persist it in the HDF5 alongside group_ids so a reloaded
  brick does not recompute.
  Risk: low; it is a pure refactor of a bounding-box computation. Watch the empty-group case
  (group_npix == 0 must still set rejected = True).
  Test: assert the precomputed bbox/count equals the current per-group computation for every
  group in one brick.

Effort: hours
```

```
#11 — Five of six copies of the auto-build fallback catch an exception that read_hdf5 never raises
Severity:    High
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/__init__.py:490-494 (and :387, :541-542, :603-605, :637-638) vs :326-330
Found by:    Seat 3

What's wrong
  read_hdf5 raises RuntimeError for a missing brick file. Five call sites catch
  (IOError, FileNotFoundError), which RuntimeError is not a subclass of — so the "build it from
  the mosaics instead" fallback is unreachable and the RuntimeError propagates out of the public
  API. One copy, in update_bricks, catches RuntimeError correctly and even carries a comment
  explaining why. The fix was applied to one copy of six.

Evidence
  farmer/image.py:3186-3187
          if not os.path.exists(path):
              raise RuntimeError(f'Cannot find file at {path}!')
  farmer/__init__.py:490-494   (detect_sources)
          try:
              brick = load_brick(brick_id)
          except (IOError, FileNotFoundError) as e:
              logger.warning(f'Could not load brick {brick_id} ({e}). Building a new brick from mosaics...')
              brick = build_bricks(brick_id, bands='detection')
  farmer/__init__.py:326-330   (update_bricks -- the one that is right)
                  try:
                      brick = load_brick(brick_id, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG'))
                  except RuntimeError as e:
                      # read_hdf5 raises RuntimeError for a brick that isn't built yet
                      logger.warning(f'Skipping brick #{brick_id}: {e}')
  Same broken idiom at :387 (detect_sources_lite), :541-542 (generate_models), :603-605
  (photometer), :637-638 (quick_group).

Impact
  Every documented entry point that advertises "loads or builds the brick" crashes instead of
  building when the brick file is absent — which is the normal state on a first run. The user is
  told to run build_bricks first (and the example script does), so this is usually masked; it
  fires when a multi-brick production loop hits one brick that failed to build earlier, taking
  down the whole loop instead of rebuilding that brick.

Fix
  Catch (RuntimeError, IOError, FileNotFoundError) in all five, or better: make read_hdf5 raise
  FileNotFoundError, which is what it actually means, and leave the five call sites alone. The
  second option is one line and fixes all five at once — but check nothing else depends on the
  RuntimeError type first (grep shows nothing does).
  Risk: FileNotFoundError is an OSError subclass, so `except (IOError, FileNotFoundError)` catches
  it. Low.
  Test: delete a brick file, call farmer.detect_sources(brick_ids=that_id), assert it rebuilds.

Effort: minutes
```

```
#12 — sys.exit() in library code kills the interpreter when a brick legitimately has no detections
Severity:    High
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/image.py:542-544
Found by:    Seat 3

What's wrong
  _extract calls sys.exit() when sep returns an empty catalog. That raises SystemExit, which
  inherits from BaseException, not Exception — so none of the pipeline's `except Exception`
  handlers catch it, IGNORE_FAILURES cannot contain it, and a multi-brick loop terminates. The
  log message itself says the condition "May be OK".

Evidence
  farmer/image.py:542-544
          if len(catalog) == 0:
              self.logger.error('No objects found! Check overlap of mosaic with this brick. May be OK. Exiting...')
              sys.exit()

Impact
  An empty brick is entirely normal at a survey edge or in a masked region — build_bricks already
  has a skiplist for exactly this case (__init__.py:198-206). A single empty brick in the middle
  of `farmer.generate_models()` over the full grid kills the process, losing every brick processed
  since the last write. Under a batch scheduler it looks like a clean exit with status 0.

Fix
  Return an empty catalog and let the caller decide:
      if len(catalog) == 0:
          self.logger.warning('No objects found -- returning an empty catalog for this brick.')
          return Table(), segmap
  then have Brick.extract short-circuit and mark the brick empty, mirroring the skiplist that
  build_bricks already maintains.
  Risk: callers currently assume a non-empty catalog — extract() at brick.py:403-417 will need an
  early return, and identify_groups/transfer_maps must tolerate zero sources. dilate_and_group
  already handles the empty case (utils.py:347-349).
  Test: run detect_sources on a brick whose detection cutout is all zeros; assert it returns and
  the loop continues to the next brick.

Effort: hours
```

```
#13 — The PSF FITS file is re-read from disk on every get_psfmodel call, roughly 30 times per group
Severity:    Medium
Category:    Performance
Verdict:     CONFIRMED (adversarially verified, both parties measured independently)
Location:    farmer/image.py:335-345, called from :713
Found by:    Seat 2

What's wrong
  get_psfmodel does fits.getdata + bad-pixel clean + float32 cast + renormalisation on every call,
  with no memoisation anywhere in the package (grep for lru_cache/functools.cache across farmer/
  finds only brick.py's `from functools import partial`). stage_images calls it once per band, and
  stage_images is called once per decision-tree stage plus once inside every measure_stats ->
  build_all_images.

Evidence
  farmer/image.py:335-340
          elif psf_path.endswith('.fits'):
              img = fits.getdata(psf_path)
              img[(img<1e-31) | np.isnan(img)] = 1e-31
              img = img.astype('float32')
              psfmodel = PixelizedPSF(img)
              self.logger.debug(f'PSF model for {band} identified as PixelizedPSF.')
  This is the shipped path: utils.py:524 stores psflist as a path string with psfcoords='none',
  and all four configured bands use a single .fits PSF.
  Call count per group, by reading the code: determine_models = stage_images:1294 +
  measure_stats:1323 (per stage) + measure_stats:1342 (final) ~ 6 x 3 MODEL_BANDS = 18;
  force_models = stage_images:1163 + measure_stats:1171 + measure_stats:1258 = 3 x 4 bands = 12.
  ~30 per group.
  Measured (warm page cache): 0.33 / 0.25 / 0.31 ms at 51 / 101 / 301 px stamps — size-independent,
  so header parsing dominates. All other numpy work in stage_images on a ~50 px group stamp:
  0.026 ms/band. The PSF read is ~11x everything else in the function it sits in.
  Current cost:  30 x 0.3 ms = 9 ms/group -> 1.5 min per 10 000-group brick.
  Proposed cost: ~0 (one read per band per process).
  What would make this wrong: a cold page cache or a network filesystem, which would make it
  considerably worse, not better; or PSF stamps large enough that pixel work dominates header
  parsing, which the measurements above rule out up to 301 px.

Impact
  ~1.5 min/brick of pure redundant I/O, plus the same allocation churn. Modest in absolute terms
  but the fix is four lines and carries essentially no risk, which is why it ranks here.

Fix
      @functools.lru_cache(maxsize=32)
      def _load_psf_image(path):
          img = fits.getdata(path); img[(img<1e-31)|np.isnan(img)] = 1e-31
          return img.astype('float32')
  and construct PixelizedPSF from a copy of the cached array.
  Risk: real, and it is the reason to copy. conf.RENORM_PSF mutates psfmodel.img *in place*
  (image.py:348) — sharing a cached array between PixelizedPSF instances would renormalise it
  repeatedly, shrinking the PSF by a further factor each time. Cache the raw array; copy on
  construction; or fold the renormalisation into the cached loader.
  Test: assert np.sum(get_psfmodel(b).img) == RENORM_PSF after ten successive calls.

Effort: minutes
```

```
#14 — The dilation structuring element is an even-sized truncated disc, so grouping dilation is off-centre
Severity:    Medium
Category:    Correctness
Verdict:     CONFIRMED (verified empirically)
Location:    farmer/utils.py:329-332 (with farmer/utils.py:614-623)
Found by:    Seat 1 and Seat 5, independently

What's wrong
  create_circular_mask(2r, 2r, radius=r) builds a 2r x 2r array with its circle centred at
  (int(w/2), int(h/2)) = (r, r) — which is not the centre of a 2r-wide array (that is r-0.5).
  The disc is clipped on the low-index side and the mask's centre of mass sits about 0.4 px toward
  +y/+x. binary_dilation with an even-sized, off-centre structuring element shifts the dilated
  mask, and scipy's origin convention then biases it in one direction.

Evidence
  farmer/utils.py:331
          struct2 = create_circular_mask(2*radius, 2*radius, radius=radius)
  farmer/utils.py:614-615
      if center is None: # use the middle of the image
          center = [int(w/2), int(h/2)]
  Measured centroid vs the true geometric centre of the array:
      radius=2  shape=(4,4)   centroid=(1.82,1.82)  geometric centre=1.50
      radius=3  shape=(6,6)   centroid=(2.89,2.89)  geometric centre=2.50
      radius=5  shape=(10,10) centroid=(4.94,4.94)  geometric centre=4.50
  With the shipped DILATION_RADIUS = 0.2 arcsec at 0.168"/px, radius_rpx rounds to 1
  (brick.py:441-443), giving a 2x2 structuring element — the degenerate worst case.

Impact
  Group membership is decided by whether dilated segments touch. An asymmetric, sub-pixel-shifted
  dilation makes that decision direction-dependent: two sources separated along one diagonal merge
  into a group while an identically-separated pair along the other does not. Group composition
  determines what is fitted jointly, so this propagates into deblended fluxes for marginal pairs.
  The effect is at the one-pixel level and only matters for pairs right at the merge threshold —
  bounded, but systematic rather than random, and it is exactly the population where deblending
  decisions matter most.

Fix
      struct2 = create_circular_mask(2*radius + 1, 2*radius + 1, radius=radius)
  which is symmetric and correctly centred. Consider scipy.ndimage.generate_binary_structure or an
  explicit iterations= dilation instead.
  Risk: group assignments change for marginal pairs, so catalogs before and after are not directly
  comparable. That is a correction, but it must be announced.
  Test: pin it. dilate_and_group on a hand-built segmap with two sources placed symmetrically
  about the centre, at +dx and -dx, must return the same group structure for both — it currently
  does not. This is a five-line test with no fixture (see #20).

Effort: minutes (fix) + hours (assessing the catalog delta)
```

```
#15 — Four copies of the same logic have drifted apart, and three of the four now behave differently
Severity:    Medium
Category:    Practice
Verdict:     CONFIRMED
Location:    see the table below
Found by:    Seat 4 and Seat 1; the individual divergences confirmed by Lead

What's wrong
  image.py is 3 344 lines and BaseImage carries detection, staging, optimisation, statistics,
  plotting and three output formats. The predictable consequence is that shared logic was copied
  rather than factored, and the copies have since been fixed or changed one at a time. This is
  not a style complaint: three of these four divergences are behavioural, and two are already
  reported above as their own findings.

Evidence
  (a) load_brick fallback, 6 copies -- __init__.py:326 catches RuntimeError; :387, :492, :542,
      :605, :638 catch (IOError, FileNotFoundError). Only one works. -> finding #11.
  (b) measure_stats ndata, 2 copies -- image.py:1575 (group level):
          ndata = np.sum(self.images[band].invvar[groupmap[0], groupmap[1]] > 0) # number of pixels
      image.py:1688 (source level):
          ndata = len(segmap[source_id][0]) # number of pixels
      The group version counts only pixels with positive inverse variance; the source version
      counts every segment pixel including masked and zero-weight ones. chi is zero at those
      pixels, so the source-level reduced chi2 is diluted low in proportion to the masked
      fraction of the segment. That statistic -- model_tracker[sid][stage]['total']['rchisq'] --
      is what decision_tree compares against SUFFICIENT_THRESH (image.py:1381, 1415-1418,
      1450-1452). Sources near a masked region or a chip gap therefore look better-fit than they
      are and are more likely to be solved as a simpler model.
  (c) utils.py:1774-1832 spawn_and_run_group is a 59-line fork of _run_group_inline
      (utils.py:1836-1889) that nothing calls -- verified dead by grep over the whole repo. It has
      drifted: it does not log completion, and because it bypasses run_group it honours neither
      conf.GROUP_TIMEOUT nor conf.IGNORE_FAILURES.
  (d) plot_image show-catalog overlay, 5 near-identical copies at image.py:2167, 2233, 2250,
      2267, 2309, with the brick-outline rectangle beside them drifted too: image.py:2162 divides
      the size by the pixel scale, image.py:2229 does not --
          ax.add_patch(Rectangle(brick_buffer_pix, self.size[0].value, self.size[1].value,
      passing degrees where pixels are required, drawing a ~0.1-pixel rectangle. Cosmetic, but it
      is the tell.

Impact
  (b) is a real, if second-order, bias in model selection. (a) is finding #11. (c) is 59 lines of
  dead code that will be copied again by whoever finds it first. (d) is a broken diagnostic. The
  compounding cost is that every future fix has to be applied N times and the author has already
  demonstrably missed some.

Fix
  Extract, in this order of payoff:
    1. `_load_or_build_brick(brick_id, bands=None)` -- one function, six call sites (fixes #11).
    2. `_chi_stats(chi, invvar, pixels, nparam)` returning the whole stats dict -- one
       implementation for both the group and source loops in measure_stats, which forces a single
       explicit decision about what ndata means. Decide in favour of the invvar>0 count.
    3. Delete spawn_and_run_group.
    4. `_overlay_catalog(ax, catalog, wcs, style)` -- one function, five call sites.
  Risk: (2) changes reported rchisq and therefore model selection. Do it deliberately, and
  re-run one brick before and after to quantify how many sources change model type.
  Test: (2) is the one that needs a pin -- assert _chi_stats on a hand-built chi image with a
  known masked fraction gives the ndof you expect.

Effort: day+
```

```
#16 — validate() checks almost nothing, so a misconfigured survey fails hours into the run
Severity:    Medium
Category:    Feature
Verdict:     CONFIRMED
Location:    farmer/__init__.py:88-93
Found by:    Seat 5 (mechanism partly corrected -- see below)

What's wrong
  validate() instantiates a Mosaic per band with load=False. That checks three things: the science
  file exists, its WCS parses, and validate_psfmodel does not raise. Everything else that can be
  wrong in a survey configuration is discovered later, in some cases many hours later.

Evidence
  farmer/__init__.py:88-93
      logger.info('Validate bands...')
      Mosaic('detection', load=False)
      for band in conf.BANDS.keys():
          Mosaic(band, load=False)
      logger.info('All bands validated successfully.')
  Not checked, each with the point at which it surfaces instead:
   - weight/mask array shape vs science: never checked; Cutout2D slices whatever it is given,
     silently pairing the wrong inverse variance with the science pixels.
   - weight-map convention (invvar vs sigma vs variance): never checked and never declared
     anywhere in config.py. See Assumption 3 -- getting this wrong scales every uncertainty in the
     catalog by wgt or wgt^2 with no symptom other than a globally wrong chi2.
   - 'zeropoint' present: raises KeyError at utils.py:1494 during write_catalog -- i.e. after the
     entire brick has been fitted.
   - output directories exist: nothing in the package ever creates them. grep for
     makedirs/mkdir over farmer/ returns nothing (only bin/prep_psf.py has any). PdfPages at
     image.py:2108 and 2439 raises FileNotFoundError on a missing PATH_FIGURES; with
     IGNORE_FAILURES = True and PLOT > 0 that exception is swallowed per group by
     utils.py:1990-1992, every group is marked rejected, and the brick completes "successfully"
     with a catalog that is entirely zero-filled by finding #1.
   - validate_psfmodel's own load test is a no-op for constant PSFs: utils.py:524 sets psflist to
     a path *string*, and utils.py:531 then does `fname = str(psfmodel[1][0])`, taking the first
     *character* of that string. fname is '/', neither .psf nor .fits, so both format branches are
     skipped and nothing is ever loaded. An unreadable or corrupt PSF passes validation.

  Correction to Seat 5: it claimed a missing zeropoint is silently backfilled with the -99
  sentinel from mosaic.py:18 and corrupts forced photometry by a factor 7e-53. REFUTED -- both
  consumers read conf.BANDS[band]['zeropoint'] directly (utils.py:1494, image.py:797/803), not
  the Mosaic properties dict, so a missing key raises KeyError and the -99 never reaches the
  photometry. The defect is real but it is a late crash, not silent corruption.

Impact
  The failure mode that matters is the output-directory one, because it composes with #1 into a
  silent total loss: a full brick of fitting, no error, an all-zero catalog. The others cost hours
  of wall-clock each time a survey is misconfigured.

Fix
  Extend validate() -- it is the natural home and already exists:
      for each band: assert science/weight/mask shapes agree (read NAXIS from the header, do not
        load the arrays); assert 'zeropoint' in conf.BANDS[band]; probe-load the PSF properly.
      os.makedirs(p, exist_ok=True) for all six PATH_* and assert os.access(p, os.W_OK).
      require an explicit conf.BANDS[b]['weight_type'] in ('invvar','sigma','variance') and
        convert in _condition_band_data, rather than assuming.
  Fix utils.py:531 to `fname = str(psfmodel[1])` for the constant case.
  Risk: requiring weight_type breaks existing config files -- default it to 'invvar' with a
  one-time warning rather than an error.
  Test: point a config at a nonexistent PATH_FIGURES and assert validate() raises in under two
  seconds.

Effort: hours
```

```
#17 — Failure is unobservable: IGNORE_FAILURES swallows everything and nothing records what a run did
Severity:    Medium
Category:    Feature
Verdict:     CONFIRMED
Location:    config/config.py:117; farmer/utils.py:1986-2044; farmer/image.py:2981-3113
Found by:    Seat 5 and Seat 3

What's wrong
  This is the root cause that #1, #5, #12 and the directory case in #16 are all symptoms of. A
  group can fail for any reason; run_group catches it, logs at ERROR, returns an empty result, and
  the brick carries on. No counter is kept, no status reaches the catalog, and the output files
  carry no record of the run at all -- so there is no way, afterwards, to tell a clean brick from
  one where 40% of groups threw.

Evidence
  config/config.py:117
      IGNORE_FAILURES = True
  farmer/utils.py:1990-1992
          except Exception as exc:
              if conf.IGNORE_FAILURES:
                  msg = f'Group #{group.group_id} inline worker failed: {repr(exc)}\n{traceback.format_exc()}'
                  return _reject_and_return(msg)
  brick.absorb (brick.py:660-702) merges the empty result without noticing it is empty.
  Provenance: write_fits (image.py:3025-3026) emits a bare `fits.PrimaryHDU()` -- no code version,
  no git hash, no input filenames, no timestamp, no config. Brick.__init__ already snapshots the
  whole config into self.config (brick.py:96-99) and write_hdf5 stores it, but nothing ever reads
  it back and it never reaches the FITS or catalog products.
  utils.header_from_dict (utils.py:558-593) exists to turn exactly that dict into a FITS header,
  is called from nowhere in the repo (verified by grep), and would crash if it were: line 576 is
  `tstart = time()` where `time` is the *module* imported at utils.py:36 -- TypeError: 'module'
  object is not callable.
  Related dead config: conf.TIMEOUT = 60 (config.py:119) is read nowhere; only GROUP_TIMEOUT
  (config.py:116, default None) is.

Impact
  A production run over 8 bricks can lose an arbitrary fraction of its sources and report success.
  Combined with #1 the loss is invisible in the catalog too. And because no output records the
  code version, config, or inputs, a published catalog cannot be traced back to what produced it
  -- which for a code underpinning COSMOS2020, SHELA and H20 is the most consequential item in
  this section.

Fix
  Three small, independent pieces:
   1. Count and report. brick.process_groups accumulates n_ok / n_rejected / n_failed and logs a
      summary line; raise if the failure fraction exceeds a configurable threshold.
   2. Per-source status column -- see #1's fit_status.
   3. Provenance. Fix header_from_dict (`import time` -> `from time import time`, or use
      time.time()), then call it in write_fits and write_catalog to stamp the PrimaryHDU with the
      config, plus __version__, the git hash, the input file paths and a UTC timestamp. Note
      header_from_dict as written also silently drops any value that is not str/float/int/list --
      which includes every astropy Quantity in the config (BRICK_BUFFER, GROUP_BUFFER,
      DILATION_RADIUS, the prior widths) and the nested BANDS dict. Handle those or the snapshot
      is misleading.
   4. Delete conf.TIMEOUT or wire it up.
  Risk: none of these change any measured number.
  Test: run a brick with a deliberately unwritable PATH_FIGURES; assert the run raises or reports
  a non-zero failure count rather than exiting 0.

Effort: hours
```

```
#18 — store_models reads a loop variable outside the loop that binds it
Severity:    Medium
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/image.py:1089-1097
Found by:    Seat 3, and Lead independently

What's wrong
  `substat` is bound by the inner `for substat in ...` loop inside the `if` branch, and then read
  in the `elif` branch of the *outer* `if`, where it is either stale from a previous outer
  iteration or unbound. In practice the outer dict iterates bands first and 'total' last, and the
  last key of a band's stats dict is 'flag' (set at image.py:1736), so the elif condition is
  always False and the intended `total_*_nomodel` statistics are simply never written. If no band
  made it into the tracker, 'total' is the first key and the line raises UnboundLocalError.

Evidence
  farmer/image.py:1089-1097
                  for stat in self.model_tracker[source_id][low_idx]:
                      if stat in self.bands:
                          for substat in self.model_tracker[source_id][low_idx][stat]:
                              if substat.endswith('chisq'):
                                  self.model_catalog[source_id].statistics[stat][f'{substat}_nomodel'] = \
                                                  self.model_tracker[source_id][low_idx][stat][substat]
                      elif substat.endswith('chisq'):
                          self.model_catalog[source_id].statistics[f'{stat}_nomodel'] = \
                                                      self.model_tracker[source_id][low_idx][stat]
  Triggering condition for the crash: model_tracker[sid][10] contains no band key. measure_stats
  `continue`s past a band that is not in self.images (image.py:1670-1672) or for which the source
  has no segmap entry (image.py:1675-1677). A source missing from the segmap in *every* staged
  band therefore reaches store_models with only 'total' in its stage-10 dict. Given finding #5
  (a weightless band is dropped) and the segmap-transfer path, that is reachable.
  Consequence when it fires: UnboundLocalError inside force_models -> caught by run_group ->
  IGNORE_FAILURES -> the whole group is rejected -> every source in it is zero-filled by #1.

Impact
  Two effects. The silent one: `total_chisq_nomodel` and friends -- the pre-forced-photometry
  reference chi2 the column names promise -- are never populated, in any run. The loud one: a
  single segmap-less source takes down its entire group, and the loss is invisible.

Fix
  Bind substat properly and make the intent explicit:
      for stat, statval in self.model_tracker[source_id][low_idx].items():
          if stat in self.bands:
              for substat, v in statval.items():
                  if substat.endswith('chisq'):
                      self.model_catalog[source_id].statistics[stat][f'{substat}_nomodel'] = v
          elif isinstance(statval, dict):
              for substat, v in statval.items():
                  if substat.endswith('chisq'):
                      self.model_catalog[source_id].statistics[f'{stat}_{substat}_nomodel'] = v
  Note the stage-10 "no model" reference is itself misnamed: force_models runs stage 10 with every
  source re-initialised as a PointSource (add_tracker at image.py:1162 seeds PointSource
  placeholders, and stage_models then builds them), so these are point-source chi2 values, not
  model-free ones. Rename or document.
  Risk: this starts writing columns that were previously absent; check write_catalog's dtype
  inference handles them (it will see floats).
  Test: force a source with no segmap in any band and assert the group still completes.

Effort: minutes
```

```
#19 — plot_image reaches for a Group-only attribute in a branch that Brick objects execute
Severity:    Medium
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/image.py:2278-2288
Found by:    Seat 4

What's wrong
  BaseImage.plot_image is shared by Mosaic, Brick and Group. The segmap/groupmap branch for
  non-detection bands indexes the groupmap dict with self.group_id, which only Group defines.

Evidence
  farmer/image.py:2278-2288
                  if imgtype in ('segmap', 'groupmap'):
                      if band != 'detection':
                      #    self.logger.warning(f'plot_image for {band} {imgtype} is NOT IMPLEMENTED YET.')
                          # get the map + identify pixels in fake image
                          groupmap = self.get_image('groupmap', band=band)
                          segmap = self.get_image('segmap', band=band)
                          source_ids = [sid for sid in segmap.keys()]
                          cmap = plt.get_cmap('rainbow', len(source_ids))
                          img = self.get_image('mask', band=band).copy().astype(np.int16)  #[src]
                          y, x = self.get_image(band=band, imgtype='groupmap')[self.group_id]
  Triggering condition: conf.PLOT > 2 and a brick that already has transferred maps. build_bricks
  (__init__.py:212-213) and update_bricks (__init__.py:334-335) both call
  brick.plot_image(show_catalog=False, show_groups=False) with imgtype=None, which iterates every
  key in self.data[band] -- including 'segmap' and 'groupmap' once detect_sources has run and the
  brick has been written and reloaded. Brick has no group_id -> AttributeError.
  (Brick.detect_sources at brick.py:577 passes imgtype='science' explicitly, so that call is safe
  -- which is why this only fires on the second visit to a brick.)

Impact
  farmer.update_bricks() with PLOT > 2 on an already-detected brick raises AttributeError.
  Diagnostics-only, and PLOT > 2 is not the default (config.py:7 sets PLOT = 0), but it fires
  exactly when a user turns plotting up to debug something else.

Fix
  Guard the branch: `if self.type == 'group':` for the group-highlight overlay, and give
  Brick/Mosaic a plain per-source rendering of the segmap dict (or skip with a debug message).
  Risk: none; it is a plotting path.
  Test: brick.plot_image() with PLOT=3 on a detected brick returns without raising.

Effort: minutes
```

```
#20 — There is no test of any kind in the repository
Severity:    Medium
Category:    Feature
Verdict:     CONFIRMED
Location:    repo-wide (bin/tractor_test.py is a standalone Tractor demo importing nothing from farmer)
Found by:    Seat 5

What's wrong
  No pytest, no unittest, no CI, no known-answer fixture. Every finding in this review that is a
  wrong *number* rather than a crash -- #1, #2, #6, #8, #14 -- would have been caught by a handful
  of pinned assertions on pure functions, with no data fixture at all.

Evidence
  find . -name '*test*' outside .git/.claude returns only bin/tractor_test.py, whose imports are
  `from tractor import *` and which references nothing in farmer/.
  No .github/workflows, no tox.ini, no pytest section in pyproject.toml.

Impact
  This is the highest-value single addition to the repo, and I want to be concrete about why
  rather than assert it. Five of the confirmed findings above are silent wrong-number bugs in
  functions that take arrays and return arrays -- no I/O, no Tractor, no sep. Each is a five-line
  assertion:
    get_fwhm(gaussian(sigma=4)) ~= 9.42            catches #8 (currently returns 1.0)
    dilate_and_group on a segmap with two sources placed symmetrically about the centre must
      give the same grouping for +dx and -dx                       catches #14
    get_params(ExpGalaxy with known PA east of north)['pa'] == that PA   catches #2
    load_brick_position(1..8) round-trips through the WCS to the expected pixel box
    clean_catalog on a hand-built mask keeps exactly the expected rows and renumbers the segmap
      consistently
    dcoord_to_offset between two known SkyCoords equals the analytic offset
  None of these needs sep, tractor, h5py, or a data file -- numpy and astropy only. That is the
  floor, and it is an afternoon's work.
  Above that floor, the end-to-end fixture worth building is a ~512x512 synthetic mosaic in three
  bands with a Gaussian PSF and ~20 injected sources of known flux and morphology, asserting that
  photometer() recovers the input fluxes to within the injected noise. That is what would have
  caught #3 and #5.

Fix
  tests/test_pure.py with the six assertions above; add pytest to a [project.optional-dependencies]
  dev extra; a three-line GitHub Actions workflow. Then tests/test_endtoend.py with the synthetic
  fixture, marked slow.
  Risk: several assertions will fail on first write. That is the finding, not a problem with the
  test.

Effort: hours (the pure-function floor); day+ (the end-to-end fixture)
```

```
#21 — brick_has_band fully deserializes the brick file to test one dictionary key
Severity:    Medium
Category:    Performance
Verdict:     CONFIRMED (adversarially verified)
Location:    farmer/__init__.py:221-222 (docstring), 242-249
Found by:    Seat 2 and Seat 3, independently

What's wrong
  The docstring promises it "reads only the top-level metadata... Much faster than loading the
  full brick". It calls recursively_load_dict_contents_from_group with the default path='/', which
  is the same call read_hdf5 uses -- it materialises every image array, catalog and model in the
  file. It then also swallows every possible error, so a genuinely corrupt brick reports "band
  absent" and is silently rebuilt.

Evidence
  farmer/__init__.py:242-249
      try:
          with h5py.File(path, 'r') as hf:
              attr = recursively_load_dict_contents_from_group(hf)
              if 'bands' in attr:
                  return band in attr['bands']
          return False
      except (IOError, FileNotFoundError, Exception):
          return False
  recursively_load_dict_contents_from_group (utils.py:1185) has no early exit and recurses generic
  groups via the else branch at 1307. self.bands is a top-level attribute (image.py:61), so the
  cheap read the docstring promises -- hf['bands'][...] -- is available.
  Caller trace: __init__.py:205 in the multi-brick production branch (reached whenever brick_ids
  is None, which bin/example_script.py documents as the production run), and __init__.py:321 in
  update_bricks. On a fresh run brick_has_band returns False and load_brick at :209 then re-reads
  the identical file: exactly 2x the brick read volume, independent of brick size.
  Note `except (IOError, FileNotFoundError, Exception)` is just `except Exception` -- the first
  two are subclasses and redundant.

Impact
  Doubles brick-file read volume in the production path. The absolute figure depends on brick size,
  which is derived at runtime and which I could not measure (no data/ directory), so I state the
  ratio rather than a GB number. Also masks corrupt-file errors as "band missing", causing a silent
  and expensive rebuild rather than a diagnosable failure.
  A related double-read sits one level down: utils.py:1258 evaluates item['data'][...] twice in a
  single expression --
      ans[key] = Cutout2D(item['data'][...], pos, np.shape(item['data'][...]), wcs=awcs)
  -- so every image dataset is read from HDF5 twice on every brick and group load. Hoisting it to
  a local is a one-line fix that halves that I/O.

Fix
      with h5py.File(path, 'r') as hf:
          if 'bands' not in hf: return False
          return band in np.asarray(hf['bands'][...]).astype(str).tolist()
  and narrow the except to (OSError, KeyError), letting anything else surface.
  Risk: low. Confirm the 'bands' dataset is written at the top level for every brick vintage --
  utils.py:1088-1097 shows it is.
  Test: time brick_has_band against load_brick on one brick; assert it is more than 10x faster.

Effort: minutes
```

```
#22 — RENORM_PSF = 1 silently renormalises every PSF to unit sum, and crashes on PsfEx models
Severity:    Medium
Category:    Correctness
Verdict:     PLAUSIBLE
Location:    farmer/image.py:347-349; config/config.py:110
Found by:    Seat 4

What's wrong
  The shipped default renormalises every loaded PSF stamp to sum exactly 1, discarding whatever
  aperture normalisation prepare_psf applied upstream. If the stamp is clipped -- which
  prepare_psf's clip_radius exists to do -- the true PSF has some flux outside it, and rescaling
  the truncated stamp to unit sum means the fitted flux is the flux within the stamp rather than
  the total. The code knows: it logs a warning saying so, on every single load.

Evidence
  farmer/image.py:347-349
          if conf.RENORM_PSF is not None:
              psfmodel.img *= conf.RENORM_PSF / np.nansum(psfmodel.img)
              self.logger.warning(f'PSF model has been renormalized to {conf.RENORM_PSF}. This WILL affect photometry!')
  config/config.py:110
      RENORM_PSF = 1
  Two separate problems:
   (a) Photometric. For a stamp containing fraction f of the true PSF flux, renormalising to 1
       biases every fitted flux by a factor f (~1-2% for a typical clip radius) -- a uniform
       multiplicative offset per band, i.e. a colour term if the clip fraction differs by band.
       This is a real aperture correction that is being silently absorbed rather than applied.
   (b) Mechanical. psfmodel.img only exists on PixelizedPSF. get_psfmodel's PsfEx branch
       (image.py:320-333) can return a PixelizedPsfEx, which has no .img -- so a user supplying
       a .psf file gets AttributeError, caught by IGNORE_FAILURES, and every group is rejected.
       I could not execute this: tractor is not installed on this machine (see Coverage), so (b)
       rests on reading tractor's psfex.py interface rather than on running it.

Impact
  (a) is a systematic multiplicative flux offset that the log warns about but that no output
  records. (b) makes the entire PsfEx code path unusable, silently, for anyone who supplies
  per-band .psf files -- and PsfEx is the format prep_psf and validate_psfmodel both anticipate.
  Marked PLAUSIBLE rather than CONFIRMED because (b) is unverified and because whether (a) matters
  depends entirely on how the user's PSF stamps were prepared, which is outside this repo.

Fix
  Default RENORM_PSF = None, and when it is set, log once per band (not per call) with the
  measured missing fraction: `logger.warning(f'{band}: PSF stamp sums to {s:.4f}; renormalising to
  {conf.RENORM_PSF}. Fluxes carry an implicit aperture correction of {conf.RENORM_PSF/s:.4f}.')`
  Record that factor in the output header. Guard the mechanical case:
      img = getattr(psfmodel, 'img', None)
      if img is None: raise ValueError(f'RENORM_PSF is set but the {band} PSF model has no pixel image')
  Risk: changing the default changes every flux by the clip fraction. This is a science decision
  for the author, not a mechanical fix -- flag, do not silently flip.
  Test (settles the PLAUSIBLE): install tractor, call get_psfmodel on a .psf file with
  RENORM_PSF = 1, and see whether it raises.

Effort: hours
```

```
#23 — Mosaic aliases the user's config dicts and then mutates them at runtime
Severity:    Low
Category:    Bug
Verdict:     CONFIRMED
Location:    farmer/mosaic.py:62-76; farmer/image.py:392-393
Found by:    Seat 3

What's wrong
  self.properties is bound to conf.BANDS[band] (or conf.DETECTION) itself, not a copy. The loop
  that follows converts every bool to an int in place, the default-backfill loop adds keys, and
  BaseImage.set_property's mosaic branch writes straight into it -- so estimate_background
  permanently injects runtime 'rms' and 'background' values into the user's configuration module.

Evidence
  farmer/mosaic.py:66-76
              if band == 'detection':
                  self.properties = conf.DETECTION
              else:
                  self.properties = conf.BANDS[band]
              for key in self.properties:
                  if isinstance(self.properties[key], bool):
                      self.properties[key] = int(self.properties[key]) # turn Trues/Falses into 1/0
              for key in default_properties:
                  if key not in self.properties:
                      self.properties[key] = default_properties[key]
  farmer/image.py:392-393
          if self.type == 'mosaic':
              self.properties[property] = value
  Reached on every mosaic load with backregion='mosaic' (mosaic.py:153-157), which is the shipped
  setting for all four bands and the detection image.

Impact
  conf.BANDS accumulates measured state across a session, so the config snapshot Brick takes at
  brick.py:96-99 records post-mutation values -- including a globalback measured from whichever
  mosaic happened to load last. If that snapshot is ever used for provenance (#17 proposes exactly
  that), it will be recording runtime state as configuration. Also makes the module non-idempotent
  under reimport in a notebook session. No number is currently wrong because of it.

Fix
      self.properties = dict(conf.DETECTION if band == 'detection' else conf.BANDS[band])
  Risk: none -- nothing depends on the mutation persisting (verified: no reader of conf.BANDS
  expects 'rms' or 'background').
  Test: assert conf.BANDS['hsc_i'] == its literal config value after constructing a Mosaic.

Effort: minutes
```

```
#24 — write_catalog locates each source by a full-table scan, once per source per column
Severity:    Low
Category:    Performance
Verdict:     CONFIRMED, but the impact claimed by two seats is overstated -- see below
Location:    farmer/image.py:3258, 3310
Found by:    Seat 2 and Seat 3, independently; measured by Lead

What's wrong
  Each assignment builds a boolean mask over the whole catalog to find one row. With N sources and
  C columns that is N*C mask constructions of length N.

Evidence
  farmer/image.py:3310
                  catalog[name][catalog['id'] == source_id] = value
  farmer/image.py:3258
              group_id = catalog['group_id'][catalog['id'] == source_id][0]
  I benchmarked the real pattern against an index-map rewrite (astropy Table, 40 columns):
      N=500   as-coded 0.16 s   index-map 0.02 s   speedup x10
      N=2000  as-coded 0.60 s   index-map 0.07 s   speedup x9
      N=5000  as-coded 1.69 s   index-map 0.16 s   speedup x11
  Both seats characterised this as quadratic and significant. It is quadratic, but at realistic N
  the cost is dominated by astropy's per-access column overhead, not the O(N) comparison, so the
  realised saving is ~1.5 s per brick at N=5000 -- against a brick that spends minutes to hours in
  Tractor. Reporting it as a headline performance win would have been wrong. It becomes worth
  doing on its own merits nearer N=20000 (~27 s), and it is cheap enough to be worth doing anyway.

Impact
  ~1.5 s per brick at N=5000; ~27 s at N=20000. Off the hot path.

Fix
      row_of = {int(v): i for i, v in enumerate(catalog['id'])}
      ...
      catalog[name][row_of[int(source_id)]] = value
  Also guard the two debug f-strings at image.py:3312-3314 and 3331-3334 -- they format a string
  for every source-column pair regardless of log level. `if logger.isEnabledFor(logging.DEBUG)`.
  Risk: none; assumes 'id' is unique, which brick.py:404 guarantees.
  Test: assert the rewritten version produces a byte-identical catalog on one brick.

Effort: minutes
```

```
#25 — Dead and unwired code: six items that read as working features and are not
Severity:    Low
Category:    Practice
Verdict:     CONFIRMED
Location:    see list
Found by:    Lead (Phase A), Seat 4, Seat 5

What's wrong
  Six things in the package are referenced by docs, config, or a public API but do not run. Each
  is individually small; together they mean a reader cannot trust that a named feature exists.

Evidence
  1. farmer/__init__.py:369  `from .utils import log_memory_usage` -- no such function in
     utils.py (grep: 0 definitions). detect_sources_lite raises ImportError on entry, and
     detect_sources(lite_mode=True) routes straight into it (__init__.py:481-484).
  2. farmer/utils.py:576  `tstart = time()` where time is the module (imported utils.py:36).
     header_from_dict raises TypeError if ever called. It is called from nowhere. -> #17.
  3. farmer/utils.py:1774-1832  spawn_and_run_group -- 59 lines, dead, drifted. -> #15(c).
  4. config/config.py:94-97  SUBTRACT_BW/BH/FW/FH -- documented in
     docs/source/configuration.rst:266-289, read nowhere. -> #3.
  5. config/config.py:119  TIMEOUT = 60 -- read nowhere; only GROUP_TIMEOUT is. -> #17.
  6. farmer/utils.py:1605-1645 get_detection_kernel(filter_kernel) ignores its own argument:
     line 1629 builds the path from conf.FILTER_KERNEL, not from the parameter --
         filename = os.path.join(dirname, '../config/conv_filters/'+conf.FILTER_KERNEL)
     Harmless today because image.py:530 passes conf.FILTER_KERNEL anyway, but any other caller
     is silently ignored.
  Also dead but harmless: rebuild_mosaic (__init__.py:648, raises NotImplemented), _clear_h5,
  generate_weight, generate_mask, plot_psf, get_resolution -- all zero non-definition references.
  generate_weight is the one worth keeping: it is the natural fix for #5.

Impact
  detect_sources_lite -- a documented public entry point -- is unusable. The rest is
  reader-confusion cost and, in the case of items 4 and 5, config that a user will reasonably
  believe is doing something.

Fix
  Write log_memory_usage (psutil-based, or drop the calls); fix header_from_dict's import; delete
  spawn_and_run_group, _clear_h5 and rebuild_mosaic; wire up or delete SUBTRACT_* and TIMEOUT; make
  get_detection_kernel use its parameter. Keep generate_weight and call it from #5's fix.
  Risk: none.
  Test: `python -c "import farmer; farmer.detect_sources_lite(brick_ids=1)"` should not
  ImportError.

Effort: hours
```

```
#26 — Repository and packaging hygiene: an absolute path, unpinned git dependencies, and 18 MB of built docs
Severity:    Low
Category:    Practice
Verdict:     CONFIRMED
Location:    config/config.py:14; pyproject.toml:8-19; Pipfile; docs/build/
Found by:    Seat 4

What's wrong
  Four separate reproducibility problems, listed once here rather than repeated per file.

Evidence
  1. config/config.py:14
         PATH_DATA = '/Users/jweaver/Projects/Software/the_farmer/data/'
     A committed absolute path into the author's home directory, which every other PATH_* derives
     from. Any clone must edit tracked source before it can run -- and that edit then shows up as
     a dirty working tree forever.
  2. pyproject.toml:9-10
         "astrometry.net @ git+https://github.com/dstndstn/astrometry.net",
         "tractor @ git+https://github.com/dstndstn/tractor",
     Both git dependencies are unpinned -- no tag, no commit. The two packages that define every
     number this code produces float to whatever is on the default branch at install time. For a
     code underpinning published survey catalogs, the environment that produced a given catalog
     cannot be reconstructed. Pin to a commit SHA.
  3. Pipfile and pyproject.toml disagree: Pipfile lists scipy, photutils and ipython and omits
     tractor, astrometry.net and sep's pin; pyproject omits scipy entirely although image.py:30
     imports it (`from scipy import stats, ndimage`) and utils.py:22 imports scipy.ndimage. numpy
     is likewise imported everywhere and declared nowhere. Pipfile also pins python 3.11 while
     pyproject says >=3.9.
  4. docs/build/ is committed -- 120 files, 18 MB of generated HTML including full source
     transcriptions of farmer/*.py, which are now stale relative to the code. `build/` is in
     .gitignore (line 6) but these were tracked before it was added, so the ignore has no effect.
     Also tracked: .DS_Store, config/.DS_Store, farmer/.DS_Store.
  Also, minor: farmer/version.py says '2.1.0-sdev', pyproject.toml:6 says '2.1.0'.

Impact
  (2) is the consequential one: it makes a published catalog non-reproducible, and it composes
  with #17's absence of provenance -- nothing records which tractor commit produced a result, and
  nothing could, because the install did not pin one.

Fix
  1. `PATH_DATA = os.environ.get('FARMER_DATA', './data')`, and ship config/config.py as
     config/config.example.py with the real one gitignored.
  2. Pin both git deps to a SHA: `tractor @ git+https://github.com/dstndstn/tractor@<sha>`.
  3. Add numpy and scipy to pyproject dependencies; delete Pipfile or regenerate it from
     pyproject -- do not maintain both.
  4. `git rm -r --cached docs/build .DS_Store */.DS_Store` and let ReadTheDocs build the docs.
  5. Single-source the version: read it from importlib.metadata, or have pyproject read version.py.
  Separately and once, since the mandate asks for it named rather than repeated: adopt ruff with
  `[tool.ruff] select = ["E","F","B"]` in pyproject.toml. `F821`/`F841` alone would have flagged
  #18's unbound `substat`, and `E714`/`B` would have flagged the `~bool` idiom behind #4.
  Risk: (1) and (4) change the developer workflow, not any result.

Effort: hours
```

---

## Quick Wins

Everything under an hour, in the order I would do them.

| # | Change | File:line | Why |
|---|---|---|---|
| 4 | `~np.isscalar` -> `not np.isscalar` | `farmer/image.py:524` | Unblocks detection on the shipped config |
| 8 | `np.nanmin` -> `np.nanmax` | `farmer/utils.py:447` | `nres` stops being the constant 1.0 |
| 2 | Drop the `90. * u.deg +` term | `farmer/utils.py:1461` | `pa` becomes the actual position angle |
| 7 | `return outdict` in both fast branches | `farmer/utils.py:812, 833` | Same-grid bands stop paying the slow reprojection |
| 6 | `getModelImage(model_bands.index(band))` | `farmer/image.py:1579-1580` | Per-band `rchisqmodel` stops being band 0 |
| 5 | `cutout.data[:] = 1.0` + a WARNING | `farmer/brick.py:319` | Weightless bands stop vanishing |
| 14 | `create_circular_mask(2*r+1, 2*r+1, ...)` | `farmer/utils.py:331` | Symmetric dilation |
| 9 | Swap the axis order | `farmer/mosaic.py:122-125` | Correct mosaic centre and size |
| 1 | NaN-fill new columns | `farmer/image.py:3309` | Unfit sources stop reading as 0.0 flux |
| 21 | Read `hf['bands']` only | `farmer/__init__.py:244-246` | Halves brick read volume |
| 21b | Hoist `item['data'][...]` to a local | `farmer/utils.py:1258` | Halves HDF5 read on every brick/group load |
| 13 | `lru_cache` the PSF array (copy on use) | `farmer/image.py:336` | ~1.5 min/brick |
| 16b | `fname = str(psfmodel[1])` | `farmer/utils.py:531` | PSF validation stops being a no-op |
| 23 | `dict(...)` around the config lookup | `farmer/mosaic.py:66-69` | Stop mutating the user's config |
| 25 | Delete `spawn_and_run_group` | `farmer/utils.py:1774-1832` | 59 dead, drifted lines |
| 26 | `os.makedirs(p, exist_ok=True)` in `validate()` | `farmer/__init__.py:89` | Prevents the silent all-zero-catalog run |

Nine of these change a number in the output catalog. If you take them together, catalogs produced
before and after are not comparable — worth a version bump and a note.

---

## Coverage

**Fully read, line by line:** `farmer/image.py` (3 344), `farmer/utils.py` (2 052), `farmer/brick.py`
(702), `farmer/__init__.py` (655), `farmer/mosaic.py` (297), `farmer/group.py` (280),
`config/config.py` (135), `bin/example_script.py`. Total 7 772 LOC — the whole package.

**Read but not deeply reviewed:** `image.py:2374-2870` (`plot_summary`, ~500 lines) was read for
control flow and for its consumption of `get_params`, but its plotting internals were not audited
in the detail the rest received. Two things I noticed there and did not chase to a finding:
`image.py:2423` indexes `catalog['source_id']`, a column that does not exist (the catalog column is
`id`, added at `brick.py:404`), so `plot_summary(group_id=...)` with `source_id=None` would raise
KeyError; and the `target_size` tuple is ordered `(dec, ra)` in the group branch and `(ra, dec)` in
the source branch, which is harmless only because both are collapsed to `max()` at `image.py:2541`.

**Not reviewed:** `bin/tractor_test.py` (standalone Tractor demo, imports nothing from farmer),
`bin/prep_psf.py` beyond confirming it imports a nonexistent module `farmer_local.utils` and so
cannot run at all, `docs/source/*.rst` except where checked against code behaviour, `docs/build/**`
(generated), `.claude/worktrees/**` (stale duplicate checkouts).

### What limited this review, stated plainly

**Nothing was executed against the real pipeline.** `sep`, `tractor`, `h5py`, `pathos`, `reproject`,
`regions` and `astrometry.net` are not installed on this machine, and there is no `data/` directory.
So: no test suite was run, no `cProfile` was taken, and no finding here rests on having watched the
pipeline behave. What I did instead, and what each finding's Evidence block reflects: extracted the
logic into standalone numpy/astropy snippets and ran those (this is how #1, #4, #6, #8, #9, #14 and
#24 were settled, and how #5 was reproduced end to end), and established external conventions by
reading Tractor's own source (this is how #2 was settled). Every performance number in this report
is either a measurement of an extracted snippet on this machine or arithmetic with the inputs shown.
The one number I could not obtain is the median per-group Tractor fit time, which is what would
decide whether #10 and #13 are important or merely tidy — that requires a real run.

**Adversarial verification was cut short.** The Phase C design was one independent skeptic per
candidate finding, briefed to refute and to default to refuting under uncertainty. 48 agents were
dispatched; 36 failed on an account spend limit partway through. So of 43 candidate findings from
the five seats, **7 received the full independent adversarial pass** (findings #3, #5, #10, #13, #21,
and two that it correctly cut down — see below). All seven came back CONFIRMED, several with
corrected severities and better evidence than the original seat supplied.

For the remainder I did the verification myself, by re-reading the cited code and its callers and
by running the snippet tests above. That is weaker than an independent skeptic — I am checking
findings I partly generated — so I want to be explicit about which is which. Findings verified by an
independent adversary: #3, #5, #10, #13, #21. Findings I verified myself by execution: #1, #2, #4,
#6, #8, #9, #14, #24. Findings verified by reading and tracing only: #7, #11, #12, #15, #16, #17,
#18, #19, #20, #22, #23, #25, #26. Only #22 is marked PLAUSIBLE rather than CONFIRMED, and it says
what would settle it.

### What the verification pass killed or corrected

Reported here because a review that hides its own error rate is not worth much.

- **A seat claimed a missing `zeropoint` is silently backfilled with `-99` and corrupts forced
  photometry by a factor 7e-53.** REFUTED. Both consumers read `conf.BANDS[band]['zeropoint']`
  directly (`utils.py:1494`, `image.py:797`), not the Mosaic properties dict where the `-99`
  sentinel lives. A missing key raises `KeyError`. The real defect is a late crash, folded into #16.
- **A seat derived finding #2's mechanism as a mirror-plus-rotation, "wrong except at |theta|=45°."**
  CORRECTED. The initialisation and readout legs cancel; the actual defect is a clean constant
  +90° offset. Right line, wrong mechanism, and the fix differs.
- **Two seats independently called `write_catalog`'s row lookup a significant quadratic cost.**
  DOWNGRADED after measurement: the realised saving is ~1.5 s per brick at N=5000, not the headline
  win claimed. Kept at Low with the measurement shown (#24).
- **A verifier caught its own seat inflating an impact figure** — the IRAC PSF effective area in #3
  was claimed as ~100 px; recomputed as 20.4 px. The conclusion survived; the number was corrected.
- **I told the panel `farmer/version.py` was empty.** It is not — it contains
  `__version__ = '2.1.0-sdev'` with no trailing newline, which is why `wc -l` reported 0. No finding
  rested on it, but any panel claim citing an empty version.py should be disregarded.

### Open questions only the author can answer

1. **Are the input mosaics already sky-subtracted?** This decides whether #3 is a real photometric
   bias today or a latent one. It is the single highest-value thing to check.
2. **What convention are the weight maps in?** Nothing in the code or config declares it (Assumption
   3). If they are sigma or variance rather than inverse variance, every uncertainty in every
   published catalog is wrong.
3. **What is the median per-group Tractor fit time on a real brick?** Decides the severity of #10
   and #13.
4. **Was the stage-2 SimpleGalaxy solve commented out deliberately?** `image.py:1396-1400` disables
   the branch, so SimpleGalaxy is unreachable at stage 2 and every marginally-resolved source is
   pushed through two extra Tractor fits before it can be selected at stage 4 or 5. The
   `decision_tree` docstring at `image.py:1362-1363` still documents the removed behaviour. If it
   was deliberate, the docstring needs updating; if not, this is a cost paid on every group.
5. **Has any published catalog been produced with `RENORM_PSF = 1` and clipped PSF stamps?** That
   determines whether #22 is a latent config issue or a correction already baked into released data.

---

## Implementation record

All 26 findings were implemented on 2026-08-26. Nothing was committed — the working tree
is left dirty for the author to review and commit.

### Files touched

| File | Change |
|---|---|
| `farmer/image.py` | #1 #2 #3 #4 #6 #12 #13 #15 #17 #18 #19 #22 #24 |
| `farmer/utils.py` | #2 #7 #8 #14 #15 #16 #17 #21 #25 |
| `farmer/brick.py` | #5 #10 #12 #16 #17 |
| `farmer/__init__.py` | #11 #16 #21 #25 |
| `farmer/mosaic.py` | #9 #23 |
| `farmer/group.py` | #10 |
| `config/config.py` | #3 #16 #17 #22 #25 |
| `bin/prep_psf.py` | rewritten as a working CLI (#25) |
| `pyproject.toml`, `.gitignore` | #26 |
| `docs/source/api/*.rst`, `configuration.rst`, `conf.py` | API and config docs realigned with the code |

### Corrections made while implementing

Three claims in the findings above did not survive contact with the code. They are
corrected in place rather than quietly dropped:

1. **#4 over-reached.** The finding claimed the `~`-on-a-bool idiom was also broken at
   `image.py:629`, `:659`, `:2487` and `:2605`. It is not: `A & ~B` is truth-equivalent to
   `A and not B` for every bool pair (`True & ~True` = `1 & -2` = `0`), so those four
   guards behaved correctly. Only the **standalone** `~np.isscalar(...)` was broken.
   The four sites were still rewritten to `and not` for legibility, no behaviour change.
   A genuine adjacent bug did turn up: `generate_mask` tested `'weight' in self.data[band]`
   and refused to generate a *mask* whenever a weight existed. Fixed to test `'mask'`.
2. **#16's zeropoint mechanism was wrong** (already noted at review time, repeated here):
   a missing zeropoint raises `KeyError`, it is not silently backfilled with `-99`.
3. **`get_fwhm` had a second bug** the finding did not mention: `np.nonzero` sorts only the
   first axis, so `dy[-1] - dy[0]` was not the y extent. The rewrite takes an explicit
   min/max per axis, and a new test asserts `get_fwhm(img) == get_fwhm(img.T)`.

### Deliberate deviations from the prescribed fix

- **`plot_psf`, `get_resolution` and `_clear_h5` deleted.** All three had zero references.
  (An earlier pass kept the first two on the grounds that they were documented public API;
  the author overruled that, so they are gone and the docs went with them:
  `get_resolution`'s `autofunction` directive was removed from `docs/source/api/utils.rst`,
  and `plot_psf` disappears from `baseimage.rst` automatically because that page uses
  `:members:`.) The now-orphaned `PSF_SIGMA_FACTOR` constant went too, along with a block
  of constants in `utils.py` that shadowed the `farmer.pure` re-exports with identical
  values. `rebuild_mosaic` was kept but converted from
  `RuntimeError('Not implelented yet!')` to a `NotImplementedError` that says what to do
  instead.
- **`SUBTRACT_BW/BH/FW/FH` were wired up rather than deleted.** `estimate_background` now
  uses them for photometric bands and `BACK_*` for detection — the split config.py and
  `configuration.rst` have always described.
- **`weight_type` is a new per-band config key**, defaulting to `'invvar'` with a one-time
  warning when undeclared. This makes Assumption 3 checkable instead of load-bearing.
  Conversion happens in `_condition_band_data` and is idempotent (it stamps the key back
  to `'invvar'`), because that method re-runs on every load from HDF5.
- **`RENORM_PSF` default flipped to `None`.** When set, the implied aperture correction is
  now computed, logged once per band, stored on `BaseImage.psf_aperture_correction`, and
  written into the output FITS/catalog headers.

- **`PATH_DATA` is now overridable** via the `FARMER_DATA` environment variable. An earlier
  pass also added a `config/config.example.py`; that was removed on review, because it
  duplicated 150 lines of `config.py` to change one, which is the exact drift pattern
  finding #15 is about.

### New behaviour worth knowing about

- **`fit_status` column** in every catalog: `0` fitted, `1` group rejected, `2` fit failed,
  `3` never attempted. New float columns are NaN-filled, not zero-filled.
- **`process_groups` now refuses to finish** a brick whose failure fraction exceeds
  `conf.MAX_FAILURE_FRACTION` (default 0.1), and checkpoints every
  `conf.CHECKPOINT_EVERY` groups (default 500).
- **Provenance headers** on FITS products and catalogs: version, git hash, UTC timestamp,
  input filenames, zeropoints, weight conventions, and 22 config parameters.
- **`validate()` now actually validates** and returns the problem list; pass `strict=False`
  to get the list instead of an exception.

### #20 was reverted at the author's request

The test suite, `farmer/pure.py` and the CI workflow were removed after review. The
finding stands as written -- five of the confirmed wrong-number bugs (#1, #2, #6, #8,
#14) are exactly what a handful of pinned assertions would have caught -- but testing
the numeric core required extracting it into a module that does not import tractor,
sep or `config`, because `farmer/utils.py` imports tractor at module scope and
`farmer/__init__.py` needs a `config` module on `sys.path`. The author judged that
extra module too high a price for the setup complexity it introduced, and preferred no
tests to that structure. All nine functions were moved back into `farmer/utils.py` at
their original positions; every bug fix inside them was preserved and re-verified.

If tests are ever wanted again, the blocker to solve first is the module-scope tractor
import in `utils.py` -- not the tests themselves.

### Verification performed, and its limits

`sep`, `tractor`, `h5py`, `scipy`, `pathos` and `pytest` are not installed on the machine
this work was done on, and there is no `data/` directory, so **the pipeline was never run
end to end.** What was done instead:

- Every edited file byte-compiles (`py_compile`).
- The numeric helpers were hand-executed with a stubbed scipy, both before and after the
  #20 revert: `get_fwhm` on four Gaussians, kernel symmetry at four radii, `clean_catalog`
  with and without a segmap, `dcoord_to_offset` at dec=+60, the `_soften_fracdev` round
  trip, `_identity_pixel_map` and `cumulative`. All pass.
- Targeted equivalence tests confirmed: `_identity_pixel_map` is bit-identical to the code
  it replaces and ~9x faster; the group bbox cache matches per-group rescanning exactly;
  the PSF cache does not compound `RENORM_PSF`; `weight_type` conversion is idempotent;
  provenance cards no longer collide; the `prep_psf` CLI works against a stubbed backend.
- `pa` was verified against Tractor's own `getRaDecBasis` matrix for six position angles.

**Run one brick in a real environment before trusting any of this.**

### Left for the author

1. **Pin the two git dependencies.** `pyproject.toml` still has `tractor` and
   `astrometry.net` unpinned; only you know which commits your results were validated
   against. This is the single largest remaining reproducibility gap.
2. **Untrack the committed build artifacts** (not done here, since no git write operations
   were performed):
   ```
   git rm -r --cached docs/build
   git rm --cached .DS_Store config/.DS_Store farmer/.DS_Store
   git rm --cached config/config.py   # then uncomment the .gitignore line
   rm Pipfile Pipfile.lock            # disagrees with pyproject; don't maintain both
   ```
3. **Decide the background question.** #3 now subtracts the background before fitting. If
   your mosaics were already sky-subtracted upstream, `globalback ≈ 0` and this is a no-op —
   worth confirming on one brick so you know which regime you are in.
