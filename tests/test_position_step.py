"""Position finite-difference step (POS_STEP) and the on-sky RA prior (PRIOR_POS_ON_SKY).

Loads farmer/utils.py by path, with ``config`` and ``reproject`` stubbed when they are
not importable, so the EDITED module is what runs even where an older farmer is
installed (the cluster container). Needs Tractor; the recovery test also needs
``astrometry.util`` for the TAN WCS. Run with
``python -m pytest tests/test_position_step.py``.

Background: PixelizedPSF has no analytic position derivatives, so Tractor differences
every model with the step from ``RaDecPos.getStepSizes()``, hard-coded to 1e-4 deg =
0.36". At 0.05"/px that is 7.2 px, a secant across the whole PSF core, and position
fits barely move. The recovery test below reproduces that and shows the fix.
"""
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits

from tractor import (PointSource, Fluxes, FluxesPhotoCal, RaDecPos, PixelizedPSF, Image,
                     Tractor, ConstantFitsWcs, ConstantSky, EllipseESoft)
from tractor.galaxy import ExpGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.constrained_optimizer import ConstrainedOptimizer

REPO = Path(__file__).resolve().parents[1]

RA, DEC = 150.0, 66.0          # deg; EDF-N-like declination so cos(dec) = 0.41 matters
MAS_DEG = 1. / 3.6e6           # deg per mas
TRACTOR_STEP_DEG = 1e-4        # deg, RaDecPos.__init__


def _load_utils():
    """Load the repo's farmer/utils.py by path and assert it is the edited one."""
    for name in ('config', 'reproject'):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            stub = types.ModuleType(name)
            if name == 'reproject':
                stub.reproject_interp = None
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location(
        'farmer_utils_under_test', REPO / 'farmer' / 'utils.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, 'set_position_step'), \
        'loaded a farmer.utils without the position-step code -- wrong file?'
    return mod


utils = _load_utils()


def _expected_steps(step_mas):
    """Tractor's convention for an on-sky step: [s / cos(dec), s] in deg."""
    s = step_mas * MAS_DEG
    return [s / np.cos(np.deg2rad(DEC)), s]


def _models():
    """One of each model class farmer stages, at (RA, DEC)."""
    shape = lambda: EllipseESoft(0., 0., 0.)
    ps = PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.))
    exp = ExpGalaxy(RaDecPos(RA, DEC), Fluxes(vis=1.), shape())
    comp = FixedCompositeGalaxy(RaDecPos(RA, DEC), Fluxes(vis=1.), SoftenedFracDev(0.5),
                                shape(), shape())
    return [ps, exp, comp]


def _prior_terms(model):
    """{name: (mu, sigma)} of the Gaussian priors on the model's position."""
    return {name: (mu, sig) for name, _, mu, sig in model.pos.gpriors.terms}


# 1. The step is applied ---------------------------------------------------------------

def test_step_applied_to_every_model_class():
    for model in _models():
        np.testing.assert_allclose(model.pos.getStepSizes(),
                                   [TRACTOR_STEP_DEG / np.cos(np.deg2rad(DEC)), TRACTOR_STEP_DEG],
                                   rtol=1e-12, err_msg='Tractor default changed?')
        out = utils.set_position_step(model, 5 * u.mas)
        assert out is model
        np.testing.assert_allclose(model.pos.getStepSizes(), _expected_steps(5.), rtol=1e-12,
                                   err_msg=model.__class__.__name__)


def test_step_accepts_any_angle_unit_and_none_leaves_tractor():
    model = PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.))
    utils.set_position_step(model, 0.005 * u.arcsec)
    np.testing.assert_allclose(model.pos.getStepSizes(), _expected_steps(5.), rtol=1e-12)
    model = PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.))
    utils.set_position_step(model, None)
    np.testing.assert_allclose(model.pos.getStepSizes(),
                               [TRACTOR_STEP_DEG / np.cos(np.deg2rad(DEC)), TRACTOR_STEP_DEG], rtol=1e-12)


def test_step_survives_a_frozen_position():
    # PHOT_PRIORS freeze the position before the step is set. ParamList.setStepSizes
    # writes thawed slots only, so the helper must use setAllStepSizes.
    model = PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.))
    model.pos.freezeAllParams()
    utils.set_position_step(model, 5 * u.mas)
    assert model.pos.getStepSizes() == []
    np.testing.assert_allclose(model.pos.getAllStepSizes(), _expected_steps(5.), rtol=1e-12)
    model.pos.thawAllParams()
    np.testing.assert_allclose(model.pos.getStepSizes(), _expected_steps(5.), rtol=1e-12)


# 2. 'auto' resolution -----------------------------------------------------------------

def test_auto_resolves_to_the_finest_fitted_band_and_ignores_detection():
    scales = {'detection': 0.01 * u.arcsec,          # finer than anything fitted: must be ignored
              'nisp_y': 0.1 * u.arcsec,
              'vis': 0.05 * u.arcsec}
    resolved = utils.resolve_position_step('auto', 0.1, scales)
    np.testing.assert_allclose(resolved.step.to(u.mas).value, 5.0)
    assert resolved.band == 'vis'
    np.testing.assert_allclose(resolved.pixscale, 0.05)

    # Per-axis arrays in degrees, as BaseImage.pixel_scales holds them.
    scales_deg = {band: np.array([v.to(u.deg).value] * 2) * u.deg for band, v in scales.items()}
    resolved = utils.resolve_position_step('auto', 0.1, scales_deg)
    np.testing.assert_allclose(resolved.step.to(u.mas).value, 5.0)
    assert resolved.band == 'vis'

    # Only coarse bands staged: the step follows them (the position is shared).
    resolved = utils.resolve_position_step('auto', 0.1, {'detection': 0.01 * u.arcsec,
                                                         'nisp_y': 0.1 * u.arcsec})
    np.testing.assert_allclose(resolved.step.to(u.mas).value, 10.0)
    assert resolved.band == 'nisp_y'


def test_fixed_and_tractor_rules_resolve_without_bands():
    resolved = utils.resolve_position_step('tractor', 0.1, {})
    assert resolved == (None, None, None)
    resolved = utils.resolve_position_step(0.005 * u.arcsec, 0.1, {})
    np.testing.assert_allclose(resolved.step.to(u.mas).value, 5.0)
    assert resolved.band is None


def test_auto_with_nothing_fitted_raises():
    with pytest.raises(ValueError):
        utils.resolve_position_step('auto', 0.1, {})
    with pytest.raises(ValueError):
        utils.resolve_position_step('auto', 0.1, {'detection': 0.01 * u.arcsec})


# 3. Recovery ------------------------------------------------------------------------

PIXSCALE = 0.05        # arcsec / pix, VIS-like
PSF_SIGMA = 2.0        # pix
NPIX = 51              # pix, image side
SNR = 30.
TRUE_OFFSET = 1.0      # pix, along +x
N_SEEDS = 50


def _gaussian_psf(sigma, half=12):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    stamp = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return stamp / stamp.sum()


def _tan_wcs():
    from astrometry.util.util import Tan
    # CRPIX at the image centre (FITS 1-based); RA increases towards -x.
    crpix = NPIX / 2. + 0.5
    scl = PIXSCALE / 3600.   # deg / pix
    return ConstantFitsWcs(Tan(RA, DEC, crpix, crpix, -scl, 0., 0., scl, float(NPIX), float(NPIX)))


def _fit_offsets(step):
    """Median recovered x-offset (pix) over N_SEEDS noise realisations, for one step.

    Mirrors farmer's optimize(): ConstrainedOptimizer, damp 0.1, DLNP_CRIT 1e-3, at
    most 50 steps, position free with no prior.
    """
    psf = _gaussian_psf(PSF_SIGMA)
    wcs = _tan_wcs()
    cx = cy = (NPIX - 1) / 2.                          # pix, 0-based centre
    flux = SNR * np.sqrt(1. / np.sum(psf ** 2))       # counts, for matched-filter S/N = SNR

    def image(data):
        return Image(data=data, invvar=np.ones_like(data), psf=PixelizedPSF(psf), wcs=wcs,
                     photocal=FluxesPhotoCal('vis'), sky=ConstantSky(0.))

    truth = PointSource(wcs.pixelToPosition(cx + TRUE_OFFSET, cy), Fluxes(vis=flux))
    truth_image = Tractor([image(np.zeros((NPIX, NPIX)))], [truth]).getModelImage(0)

    offsets = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        data = truth_image + rng.standard_normal(truth_image.shape)   # sigma = 1 per pixel
        src = PointSource(wcs.pixelToPosition(cx, cy), Fluxes(vis=flux))
        utils.set_position_step(src, step)
        tr = Tractor([image(data)], [src])
        tr.freezeParam('images')
        tr.optimizer = ConstrainedOptimizer()
        tr.optimizer.hit_limit = False
        tr.optimizer.last_step_hit_limit = False
        for _ in range(50):
            dlnp, X, alpha, var = tr.optimize(variance=True, damp=0.1, shared_params=False)
            if dlnp < 1e-3:
                break
        x, y = wcs.positionToPixel(src.pos)
        offsets.append(x - cx)
    return float(np.median(offsets)), float(np.std(offsets))


def test_recovery_with_fixed_step_and_failure_with_tractor_step():
    resolved = utils.resolve_position_step('auto', 0.1, {'vis': PIXSCALE * u.arcsec})
    np.testing.assert_allclose(resolved.step.to(u.mas).value, 5.0)

    fixed, fixed_scatter = _fit_offsets(resolved.step)
    legacy, legacy_scatter = _fit_offsets(None)
    print(f'\nrecovered offset for a {TRUE_OFFSET:.1f} px truth at S/N {SNR:.0f}, {N_SEEDS} seeds: '
          f'step 5 mas -> median {fixed:.3f} px (scatter {fixed_scatter:.3f}); '
          f'tractor 360 mas -> median {legacy:.3f} px (scatter {legacy_scatter:.3f})')

    # Fisher error for a Gaussian PSF is sqrt(2) sigma / SNR = 0.094 px, so the median
    # over 50 seeds is good to ~0.02 px; 10% of the truth is a safe band.
    assert abs(fixed - TRUE_OFFSET) < 0.1 * TRUE_OFFSET, fixed
    assert fixed_scatter < 3 * np.sqrt(2) * PSF_SIGMA / SNR, fixed_scatter
    # With the 7.2 px secant the fit is short by more than a quarter.
    assert legacy < 0.75 * TRUE_OFFSET, legacy


# 4. Prior on sky --------------------------------------------------------------------

def test_prior_sigma_is_an_on_sky_angle():
    sigma = 0.1 * u.arcsec
    for model in _models():
        model = utils.set_priors(model, {'pos': sigma}, prior_pos_on_sky=True)
        terms = _prior_terms(model)
        assert set(terms) == {'ra', 'dec'}
        np.testing.assert_allclose(terms['ra'][1], sigma.to(u.deg).value / np.cos(np.deg2rad(DEC)),
                                   rtol=1e-12)
        np.testing.assert_allclose(terms['dec'][1], sigma.to(u.deg).value, rtol=1e-12)
        assert terms['ra'][0] == RA and terms['dec'][0] == DEC
        assert len(model.pos.getStepSizes()) == 2, 'a prior must not freeze the position'


# 5. Legacy path ---------------------------------------------------------------------

def test_legacy_config_keeps_todays_behaviour_and_warns_once():
    utils._legacy_position_warned = False
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger('farmer.priors')
    logger.addHandler(handler)
    try:
        settings = utils.position_step_settings(types.SimpleNamespace())   # neither key
        again = utils.position_step_settings(types.SimpleNamespace())
    finally:
        logger.removeHandler(handler)
    assert settings == {'pos_step': 'tractor', 'pixfrac': 0.1, 'prior_pos_on_sky': False,
                        'legacy': True}
    assert again == settings
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in records]
    assert 'legacy' in warnings[0].getMessage()

    resolved = utils.resolve_position_step(settings['pos_step'], settings['pixfrac'],
                                           {'vis': 0.05 * u.arcsec})
    assert resolved.step is None

    model = PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.))
    utils.set_position_step(model, resolved.step)
    np.testing.assert_allclose(model.pos.getStepSizes(),
                               [TRACTOR_STEP_DEG / np.cos(np.deg2rad(DEC)), TRACTOR_STEP_DEG], rtol=1e-12)

    model = utils.set_priors(model, {'pos': 0.1 * u.arcsec},
                             prior_pos_on_sky=settings['prior_pos_on_sky'])
    terms = _prior_terms(model)
    np.testing.assert_allclose(terms['ra'][1], 0.1 / 3600., rtol=1e-12)
    np.testing.assert_allclose(terms['dec'][1], 0.1 / 3600., rtol=1e-12)

    # The default argument is the legacy form too, so an untouched caller is unchanged.
    model = utils.set_priors(PointSource(RaDecPos(RA, DEC), Fluxes(vis=1.)), {'pos': 0.1 * u.arcsec})
    assert _prior_terms(model)['ra'][1] == _prior_terms(model)['dec'][1]


# Config validation and header cards ---------------------------------------------------

def test_settings_are_validated_and_one_key_opts_in():
    settings = utils.position_step_settings(
        types.SimpleNamespace(POS_STEP='auto', POS_STEP_PIXFRAC=0.1, PRIOR_POS_ON_SKY=True))
    assert settings == {'pos_step': 'auto', 'pixfrac': 0.1, 'prior_pos_on_sky': True, 'legacy': False}

    settings = utils.position_step_settings(types.SimpleNamespace(POS_STEP=0.005 * u.arcsec))
    assert settings['pos_step'] == 0.005 * u.arcsec
    assert settings['prior_pos_on_sky'] is False and settings['legacy'] is False

    settings = utils.position_step_settings(types.SimpleNamespace(PRIOR_POS_ON_SKY=True))
    assert settings['pos_step'] == 'tractor' and settings['prior_pos_on_sky'] is True
    assert settings['legacy'] is False

    for bad in ({'POS_STEP': 'fine'}, {'POS_STEP': 5.0}, {'POS_STEP': -1 * u.arcsec},
                {'POS_STEP': 1 * u.Jy}, {'POS_STEP': [1, 2] * u.arcsec}, {'POS_STEP': None},
                {'POS_STEP': 'auto', 'POS_STEP_PIXFRAC': 0}, {'POS_STEP': 'auto', 'POS_STEP_PIXFRAC': '0.1'},
                {'POS_STEP': 'auto', 'PRIOR_POS_ON_SKY': 'yes'}, {'PRIOR_POS_ON_SKY': 1}):
        with pytest.raises(ValueError):
            utils.position_step_settings(types.SimpleNamespace(**bad))


def test_header_cards_record_rule_and_resolved_step():
    settings = utils.position_step_settings(
        types.SimpleNamespace(POS_STEP='auto', POS_STEP_PIXFRAC=0.1, PRIOR_POS_ON_SKY=True))
    resolved = utils.resolve_position_step('auto', 0.1, {'vis': 0.05 * u.arcsec, 'nisp_y': 0.1 * u.arcsec})
    cards = utils.position_step_header(settings, resolved)
    assert cards['POSSTEP'][0] == 'auto'
    np.testing.assert_allclose(cards['POSSTEPM'][0], 5.0)
    assert cards['POSPIXFR'][0] == 0.1
    assert cards['PRIORSKY'][0] is True
    hdr = fits.Header()
    for key, value in cards.items():
        hdr[key] = value          # (value, comment) tuples must be valid FITS cards
    assert hdr['POSSTEPM'] == 5.0

    settings = utils.position_step_settings(types.SimpleNamespace())
    resolved = utils.resolve_position_step(settings['pos_step'], settings['pixfrac'], {})
    cards = utils.position_step_header(settings, resolved)
    assert cards['POSSTEP'][0] == 'tractor'
    assert cards['POSSTEPM'][0] == -1.0
    assert 'POSPIXFR' not in cards
    assert cards['PRIORSKY'][0] is False

    settings = utils.position_step_settings(types.SimpleNamespace(POS_STEP=0.005 * u.arcsec))
    resolved = utils.resolve_position_step(settings['pos_step'], settings['pixfrac'], {})
    cards = utils.position_step_header(settings, resolved)
    assert cards['POSSTEP'][0] == '0.005 arcsec'
    np.testing.assert_allclose(cards['POSSTEPM'][0], 5.0)
    assert 'auto -> ' not in utils.describe_position_step(settings, resolved)
