#!/usr/bin/env python
"""Pre-process PSF stamps for The Farmer.

Resamples, background-subtracts, clips and renormalises PSF FITS stamps so they
are ready to hand to the pipeline. Run it once per band, before ``farmer.validate()``.

Examples
--------
Clip and normalise a directory of HSC-I stamps, keeping 90.36% of the flux inside
a 3 arcsec radius aperture::

    python bin/prep_psf.py 'psfmodels/HSC-I/*.fits' -o psfmodels/HSC-I_proc \\
        --pixel-scale 0.17 --mask-radius 3.06 --clip-radius 3.06 \\
        --norm 0.9036244689222772 --norm-radius 3.0

Resample oversampled IRAC ch1 stamps onto the native 0.6 arcsec grid::

    python bin/prep_psf.py 'psfmodels/ch1/*oversamp*.fits' -o psfmodels/ch1_native \\
        --pixel-scale 0.012 --target-pixel-scale 0.6 \\
        --clip-radius 10 --norm 0.9371 --norm-radius 10 --rename oversamp:native

Encircled-energy fractions measured for the surveys this code was built for, for
reference -- measure your own rather than reusing these:

    HSC-R 0.8660520150634863   HSC-I 0.9036244689222772   HSC-Z 0.9223879278478824
    (at 3 arcsec radius)
    IRAC ch1 0.9371            IRAC ch2 0.9249
    (at 10 arcsec radius)
"""

import argparse
import glob
import os
import sys

import astropy.units as u

# Make `farmer` importable when this script is run straight out of a checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from farmer.utils import prepare_psf


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern',
                   help='glob pattern matching the input PSF FITS files, e.g. "psf/HSC-I/*.fits"')
    p.add_argument('-o', '--outdir', required=True,
                   help='directory to write processed stamps into (created if absent)')
    p.add_argument('--pixel-scale', type=float, default=None,
                   help='pixel scale of the input stamps, arcsec/pixel. '
                        'Read from the FITS WCS if omitted.')
    p.add_argument('--target-pixel-scale', type=float, default=None,
                   help='resample to this pixel scale, arcsec/pixel')
    p.add_argument('--mask-radius', type=float, default=None,
                   help='estimate and subtract a background plateau outside this radius, arcsec')
    p.add_argument('--clip-radius', type=float, default=None,
                   help='clip the stamp at this radius, arcsec')
    p.add_argument('--norm', type=float, default=None,
                   help='renormalise so the flux inside --norm-radius equals this value '
                        '(i.e. the encircled-energy fraction at that radius)')
    p.add_argument('--norm-radius', type=float, default=None,
                   help='aperture radius for --norm, arcsec. Required when --norm is given.')
    p.add_argument('--ext', type=int, default=0, help='FITS extension to read and write')
    p.add_argument('--rename', default=None, metavar='OLD:NEW',
                   help='substring replacement applied to each output filename')
    p.add_argument('-n', '--dry-run', action='store_true',
                   help='list what would be written, without doing it')
    args = p.parse_args(argv)

    # prepare_psf divides norm_radius by pixel_scale, so one without the other is a
    # TypeError deep inside the call. Catch it here instead.
    if (args.norm is None) != (args.norm_radius is None):
        p.error('--norm and --norm-radius must be given together')

    files = sorted(glob.glob(args.pattern))
    if not files:
        p.error(f'no files matched {args.pattern!r}')

    os.makedirs(args.outdir, exist_ok=True)     # makedirs, not mkdir: parents may be missing

    arcsec = lambda v: None if v is None else v * u.arcsec
    for fn in files:
        name = os.path.basename(fn)
        if args.rename:
            old, _, new = args.rename.partition(':')
            name = name.replace(old, new)
        outfile = os.path.join(args.outdir, name)

        if args.dry_run:
            print(f'{fn} -> {outfile}')
            continue

        prepare_psf(fn, outfile,
                    pixel_scale=arcsec(args.pixel_scale),
                    target_pixel_scale=arcsec(args.target_pixel_scale),
                    mask_radius=arcsec(args.mask_radius),
                    clip_radius=arcsec(args.clip_radius),
                    norm=args.norm,
                    norm_radius=arcsec(args.norm_radius),
                    ext=args.ext)

    print(f'\nProcessed {len(files)} PSF stamp(s) into {args.outdir}')


if __name__ == '__main__':
    main()
