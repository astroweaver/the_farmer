# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Run standard checks on the photometry

Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import adv_tools as adv
import astro_tools as astr
import astropy.units as u 

from matplotlib.colors import LogNorm
import scipy.stats

# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
fname_cat= '/Volumes/WD4/Current/tractor_pipeline/data/catalogs/master_catalog.fits'
out_dir= '/Users/jweaver/Projects/Current/COSMOS/tractor_pipeline/figures'


fname_vcat = '../../data/external/SPLASH_SXDF_Mehta+_v1.6/SPLASH_SXDF_Mehta+_v1.6.fits'
# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------
cat = Table.read(fname_cat)

vcat = Table.read(fname_vcat)

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------

# Convert to magnitudes!
flux_i = cat['hsc_i']
flux_z = cat['hsc_z']

vmag_i = vcat['MAG_AUTO_hsc_i']
vmag_z = vcat['MAG_AUTO_hsc_z']

mask = (flux_i != flux_i) & (flux_z != flux_z) | (flux_i < 0) | (flux_z < 0)
flux_i = flux_i[~mask]
flux_z = flux_z[~mask]

cat = cat[~mask]

zpt = 23.93

mag_i = - 2.5 * np.log10(flux_i) + zpt
mag_z = - 2.5 * np.log10(flux_z) + zpt


plt.ioff()

# Color-mag plot
col1 = mag_i
col2 = mag_z

vcol1 = vmag_i
vcol2 = vmag_z

x, y = cat['x'], cat['y']
vx, vy = vcat['X_IMAGE'], vcat['Y_IMAGE']

z, vz = np.zeros(len(y)), np.zeros(len(vy))

coord = SkyCoord(x, y, z, unit='pixel', representation_type='cartesian')
vcoord = SkyCoord(vx, vy, vz, unit='pixel', representation_type='cartesian')

idx, _, sep = coord.match_to_catalog_3d(vcoord)  
sep = sep.value

thresh = 1.5
count = np.sum(sep < thresh)

fig, ax = plt.subplots()
ax.hist(sep, bins=np.arange(0, max(sep), 0.1))
ax.set_xlim(0, 10)
ax.axvline(thresh, linestyle='dotted', color='k')
ax.text(0.7, 0.85, f'N = {count}', transform=ax.transAxes)
fig.savefig(os.path.join(out_dir, 'separation.png'))

idx_near = sep < thresh
vidx_near = idx[sep < thresh]

col1, col2 = col1[idx_near], col2[idx_near]
vcol1, vcol2 = vcol1[vidx_near], vcol2[vidx_near]

fig, ax = plt.subplots(ncols = 2, sharex=True, sharey=True)
ax[0].plot([0, 100], [0,100], c='grey', ls='dotted', zorder=-1)
ax[1].plot([0, 100], [0,100], c='grey', ls='dotted', zorder=-1)
ax[0].scatter(vcol1, col1, s=0.2, c='purple', alpha=0.2)
ax[0].text(0.1, 0.9, f'N = {len(col1)}', transform=ax[0].transAxes )
ax[1].scatter(vcol2, col2, s=0.2, c='purple', alpha=0.2)
# ax[1].text(0.1, 0.9 )
#ax[0].set(xlim=(16,29), ylim=(-5,5))
ax[0].set(xlim=(16,29), ylim=(16,29))
ax[0].set(xlabel='Mehta HSC i (AB)', ylabel='Tractor HSC i (AB)')
ax[1].set(xlabel='Mehta HSC z (AB)', ylabel='Tractor HSC z (AB)')

fig.subplots_adjust(bottom=0.2)

fig.savefig(os.path.join(out_dir, 'master_colcol.png'))


# DIFF

minmag, maxmag = 19, 27
miny, maxy = -0.5, 0.5

diff1 = col1 - vcol1
diff2 = col2 - vcol2

def mean_confidence_interval(data, confidence=0.34):
    m = np.median(data)
    sdata = np.sort(data)
    hdata = sdata[sdata > m]
    ldata = sdata[sdata < m]
    n_hdata = len(hdata)
    n_ldata = len(ldata)
    hmax = hdata[(np.arange(n_hdata) / n_hdata) < confidence][-1]
    hmin = ldata[::-1][(np.arange(n_ldata) / n_ldata) < confidence][-1]
    return m, hmin, hmax

def running_med(X, Y, xrange, total_bins):
    bins = np.linspace(xrange[0], xrange[1], total_bins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X,bins)
    foo = np.array([mean_confidence_interval(Y[idx==k]) for k in range(total_bins)])
    running_median, running_std = foo[:,0], np.array((foo[:,1], foo[:,2]))
    Nbins = np.array([np.sum(idx==k) for k in range(total_bins)])
    return Nbins, np.array(bins - delta/2.), running_median, running_std

total_bins = 10
Nbins1, rbins1, rmed1, rstd1 = running_med(col1, diff1, xrange=(20, 26.5), total_bins = total_bins)
Nbins2, rbins2, rmed2, rstd2 = running_med(col2, diff2, xrange=(20, 26.5), total_bins = total_bins)

rbins1, rmed1, rstd1 = rbins1[Nbins1 > 1], rmed1[Nbins1 > 1], rstd1[:, Nbins1 > 1]
rbins2, rmed2, rstd2 = rbins2[Nbins2 > 1], rmed2[Nbins2 > 1], rstd2[:, Nbins2 > 1]

fig, ax = plt.subplots(nrows = 2, sharex=True, sharey=True, figsize=(15, 10))

xbins = np.linspace(minmag, maxmag, 150)
ybins = np.linspace(miny, maxy, int(150 * minmag/maxmag))

opt = dict(bins=[xbins, ybins], range=[[minmag,maxmag], [miny, maxy]], zorder=-1, cmap='Blues', norm=LogNorm())

ax[0].axhline(0, linestyle='dashed', c='k')
cax1 = ax[0].hist2d(col1, diff1, **opt)
fig.colorbar(cax1[3], ax=ax[0])
ax[0].plot(rbins1, rmed1, c='orange')
ax[0].fill_between(rbins1, rstd1[0], rstd1[1], color='orange', alpha = 0.3)


ax[1].axhline(0, linestyle='dashed', c='k')
cax2 = ax[1].hist2d(col2, diff2, **opt)
fig.colorbar(cax2[3], ax=ax[1])
ax[1].plot(rbins2, rmed2, c='orange')
ax[1].fill_between(rbins2, rstd2[0], rstd2[1], color='orange', alpha = 0.3)

ax[0].set_xlim(minmag, maxmag)
ax[0].set_ylim(miny, maxy)
ax[0].text(0.05, 0.9, f'Confidence: 34%', transform=ax[0].transAxes)

ax[0].set(xlabel='Mehta HSC i (AB)', ylabel='Tractor - Mehta HSC i (AB)')
ax[1].set(xlabel='Mehta HSC z (AB)', ylabel='Tractor - Mehta HSC z (AB)')

fig.subplots_adjust(right=1.1)
fig.savefig(os.path.join(out_dir, 'master_diff.png'))



# Number counts
fig, ax = plt.subplots(nrows = 2, sharex = True, sharey=True)
bins = np.arange(16, 29, 0.5)
ax[0].hist(mag_i, histtype='step', bins = bins, label = f'Tractor (N = {len(mag_z)})', density=True, color='royalblue', ls='solid')
ax[1].hist(mag_z, histtype='step', bins = bins, label = 'Tractor', density=True, color='orange', ls='solid')
ax[0].hist(vmag_i, histtype='step', bins = bins, label = f'Mehta (N = {len(vmag_i)})', density=True, color='royalblue', ls='dotted')
ax[1].hist(vmag_z, histtype='step', bins = bins, label = 'Mehta', density=True, color='orange', ls='dotted')
ax[1].set(xlim=(16, 29), ylim=(0,0.4), xlabel='Mag (AB)', ylabel='                  Norm count' )

ax[0].text( 16.5, 0.32, 'HSC i')
ax[1].text( 16.5, 0.32, 'HSC z')
ax[0].legend(loc=3)
ax[1].legend(loc=3)

fig.subplots_adjust(left = 0.1, bottom=0.2)
fig.savefig(os.path.join(out_dir, 'master_numcount.png'))


fig, ax = plt.subplots()
ax.plot((mag_i - mag_z)[mag_i > 27])
fig.savefig(os.path.join(out_dir, 'check.pdf'))
