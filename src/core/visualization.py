"""

Filename: visualization.py

Purpose: Set of go-to plotting functions

Author: John Weaver
Date created: 28.11.2018
Possible problems:
1.

"""
import os
import numpy as np
from tractor.galaxy import ExpGalaxy
from tractor import EllipseE
from tractor.galaxy import ExpGalaxy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries
from astropy.visualization import hist
from scipy import stats

import config as conf
import matplotlib.cm as cm
import random
from time import time
from astropy.io import fits

import logging
logger = logging.getLogger('farmer.visualization')

# Random discrete color generator
colors = cm.rainbow(np.linspace(0, 1, 1000))
cidx = np.arange(0, 1000)
random.shuffle(cidx)
colors = colors[cidx]

def plot_background(brick, idx, band=''):
    fig, ax = plt.subplots(figsize=(20,20))
    vmin, vmax = brick.background_images[idx].min(), brick.background_images[idx].max()
    vmin = -vmax
    img = ax.imshow(brick.background_images[idx], cmap='RdGy', norm=SymLogNorm(linthresh=0.03))
    # plt.colorbar(img, ax=ax)
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_background.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    logger.info(f'Saving figure: {out_path}')

def plot_mask(brick, idx, band=''):
    fig, ax = plt.subplots(figsize=(20,20))
    
    img = ax.imshow(brick.masks[idx])
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_mask.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    logger.info(f'Saving figure: {out_path}')

def plot_brick(brick, idx, band=''):
    fig, ax = plt.subplots(figsize=(20,20))
    backlevel, noisesigma = brick.backgrounds[idx]
    vmin, vmax = np.max([backlevel + noisesigma, 1E-5]), brick.images[idx].max()
    # vmin, vmax = brick.images[idx].min(), brick.images[idx].max()
    if vmin > vmax:
        logger.warning(f'{band} brick not plotted!')
        return
    vmin = -vmax
    norm = SymLogNorm(linthresh=0.03)
    img = ax.imshow(brick.images[idx], cmap='RdGy', origin='lower', norm=norm)
    # plt.colorbar(img, ax=ax)
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_brick.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    logger.info(f'Saving figure: {out_path}')

def plot_blob(myblob, myfblob):

    fig, ax = plt.subplots(ncols=4, nrows=1+myfblob.n_bands, figsize=(5 + 5*myfblob.n_bands, 10), sharex=True, sharey=True)
        
    back = myblob.backgrounds[0]
    mean, rms = back[0], back[1]
    noise = np.random.normal(mean, rms, size=myfblob.dims)
    tr = myblob.solution_tractor
    
    norm = LogNorm(np.max([mean + rms, 1E-5]), myblob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)
    # img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

    mmask = myblob.masks[0].copy()
    mmask[mmask==1] = np.nan
    ax[0, 0].imshow(myblob.images[0], **img_opt)
    ax[0, 0].imshow(mmask, alpha=0.5, cmap='Greys')
    ax[0, 1].imshow(myblob.solution_model_images[0] + noise, **img_opt)
    ax[0, 2].imshow(myblob.images[0] - myblob.solution_model_images[0], cmap='RdGy', vmin=-5*rms, vmax=5*rms)    
    ax[0, 3].imshow(myblob.solution_chi_images[0], cmap='RdGy', vmin = -7, vmax = 7)
    
    ax[0, 0].set_ylabel(f'Detection ({myblob.bands[0]})')
    ax[0, 0].set_title('Data')
    ax[0, 1].set_title('Model')
    ax[0, 2].set_title('Data - Model')
    ax[0, 3].set_title('$\chi$-map')
    
    band = myblob.bands[0]
    for j, src in enumerate(myblob.solution_catalog):
        try:
            mtype = src.name
        except:
            mtype = 'PointSource'
        flux = src.getBrightness().getFlux(band)
        chisq = myblob.solved_chisq[j]
        topt = dict(color=colors[j], transform = ax[0, 3].transAxes)
        ystart = 0.99 - j * 0.4
        ax[0, 3].text(1.05, ystart - 0.1, f'{j}) {mtype}', **topt)
        ax[0, 3].text(1.05, ystart - 0.2, f'  F({band}) = {flux:4.4f}', **topt)
        ax[0, 3].text(1.05, ystart - 0.3, f'  $\chi^{2}$ = {chisq:4.4f}', **topt)

        objects = myblob.bcatalog[j]
        e = Ellipse(xy=(objects['x'], objects['y']),
                    width=6*objects['a'],
                    height=6*objects['b'],
                    angle=objects['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax[0, 0].add_artist(e)

    try:
        for i in np.arange(myfblob.n_bands):
            back = myfblob.backgrounds[i]
            mean, rms = back[0], back[1]
            noise = np.random.normal(mean, rms, size=myfblob.dims)
            tr = myfblob.solution_tractor
            
            # norm = LogNorm(np.max([mean + rms, 1E-5]), myblob.images.max(), clip='True')
            # img_opt = dict(cmap='Greys', norm=norm)
            img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

            ax[i+1, 0].imshow(myfblob.images[i], **img_opt)
            ax[i+1, 1].imshow(myfblob.solution_model_images[i] + noise, **img_opt)
            ax[i+1, 2].imshow(myfblob.images[i] - myfblob.solution_model_images[i], cmap='RdGy', vmin=-5*rms, vmax=5*rms)    
            ax[i+1, 3].imshow(myfblob.solution_chi_images[i], cmap='RdGy', vmin = -7, vmax = 7)
            
            ax[i+1, 0].set_ylabel(myfblob.bands[i])
            
            band = myfblob.bands[i]
            for j, src in enumerate(myfblob.solution_catalog):
                try:
                    mtype = src.name
                except:
                    mtype = 'PointSource'
                flux = src.getBrightness().getFlux(band)
                chisq = myfblob.solution_chisq[j, i]
                Nres = myfblob.n_residual_sources[i]
                topt = dict(color=colors[j], transform = ax[i+1, 3].transAxes)
                ystart = 0.99 - j * 0.4
                ax[i+1, 3].text(1.05, ystart - 0.1, f'{j}) {mtype}', **topt)
                ax[i+1, 3].text(1.05, ystart - 0.2, f'  F({band}) = {flux:4.4f}', **topt)
                ax[i+1, 3].text(1.05, ystart - 0.3, f'  $\chi^{2}$ = {chisq:4.4f}', **topt)
                if Nres > 0:
                    ax[i+1, 3].text(1.05, ystart - 0.4, f'{Nres} residual sources found!', **topt)
                    
                    res_x = myfblob.residual_catalog[i]['x']
                    res_y = myfblob.residual_catalog[i]['y']
                    for x, y in zip(res_x, res_y):
                        ax[i+1, 3].scatter(x, y, marker='+', color='r')
            
        for s, src in enumerate(myfblob.solution_catalog):
            x, y = src.pos
            color = colors[s]
            for i in np.arange(1 + myfblob.n_bands):
                for j in np.arange(4):
                    ax[i,j].plot([x, x], [y - 10, y - 5], c=color)
                    ax[i,j].plot([x - 10, x - 5], [y, y], c=color)
    except:
        logger.warning('Could not plot multiwavelength diagnostic figures')
                
    
    [[ax[i,j].set(xlim=(0,myfblob.dims[1]), ylim=(0,myfblob.dims[0])) for i in np.arange(myfblob.n_bands+1)] for j in np.arange(4)]
        
    #fig.suptitle(f'Solution for {blob_id}')
    fig.subplots_adjust(wspace=0.01, hspace=0, right=0.8)
    if myblob._is_itemblob:
        sid = myblob.bcatalog['source_id'][0]
        fig.savefig(os.path.join(conf.PLOT_DIR, f'{myblob.brick_id}_B{myblob.blob_id}_S{sid}.pdf'))
    else:
        fig.savefig(os.path.join(conf.PLOT_DIR, f'{myblob.brick_id}_B{myblob.blob_id}.pdf'))
    plt.close()

def plot_srcprofile(blob, src, sid, bands=None):

    if bands is None:
        band = conf.MODELING_NICKNAME
        nickname = conf.MODELING_NICKNAME
        bidx = [0,]
        bands = [band,]
        outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_srcprofile.pdf')
    elif (len(bands) == 1) & (bands[0] == conf.MODELING_NICKNAME):
            band = conf.MODELING_NICKNAME
            nickname = conf.MODELING_NICKNAME
            bidx = [0,]
            bands = [band,]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_srcprofile.pdf')
    else:
        bidx = [blob._band2idx(b, bands=blob.bands) for b in bands]
        if bands[0].startswith(conf.MODELING_NICKNAME):
            nickname = conf.MODELING_NICKNAME
        else:
            nickname = conf.MULTIBAND_NICKNAME
        if len(bands) > 1:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{nickname}_srcprofile.pdf')
        else:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{bands[0]}_srcprofile.pdf')
        


    import matplotlib.backends.backend_pdf
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath)
    
    for idx, band in zip(bidx, bands):
        
        if band == conf.MODELING_NICKNAME:
            zpt = conf.MODELING_ZPT
            rband = conf.MODELING_NICKNAME
        elif band.startswith(conf.MODELING_NICKNAME):
            band_name = band[len(conf.MODELING_NICKNAME)+1:]
            # zpt = conf.MULTIBAND_ZPT[blob._band2idx(band_name)]
            if band_name == conf.MODELING_NICKNAME:
                zpt = conf.MODELING_ZPT
                rband = band
            else:
                zpt = conf.MULTIBAND_ZPT[blob._band2idx(band_name)]
                rband = conf.MODELING_NICKNAME + '_' + band_name
        else:
            zpt = conf.MULTIBAND_ZPT[blob._band2idx(band)]
            rband = conf.MODELING_NICKNAME + '_' + band

        # information
        bid = blob.blob_id
        bsrc = blob.bcatalog[blob.bcatalog['source_id'] == sid]
        ra, dec = bsrc['RA'][0], bsrc['DEC'][0]
        if nickname == conf.MODELING_NICKNAME:
            xp0, yp0 = bsrc['x_orig'][0] - blob.subvector[1], bsrc['y_orig'][0] - blob.subvector[0]
        else:
            xp0, yp0 = bsrc['x_orig'][0] - blob.subvector[1] - blob.mosaic_origin[1] + conf.BRICK_BUFFER, bsrc['y_orig'][0] - blob.subvector[0] - blob.mosaic_origin[0] + conf.BRICK_BUFFER
        xp, yp = src.pos[0], src.pos[1]

        xps, yps = xp, yp
        flux, flux_err = bsrc[f'FLUX_{band}'][0], bsrc[f'FLUXERR_{band}'][0]
        mag, mag_err = bsrc[f'MAG_{band}'][0], bsrc[f'MAGERR_{band}'][0]
        n_blob = bsrc['N_BLOB'][0]
        chi2 = bsrc[f'CHISQ_{band}'][0]
        snr = bsrc[f'SNR_{band}'][0]

        is_resolved = False
        if src.name not in ('PointSource', 'SimpleGalaxy'):
            is_resolved = True
            col = np.array(bsrc.colnames)[np.array([tcoln.startswith('REFF') for tcoln in bsrc.colnames])][0]
            rband = col[len('REFF_'):]

            reff, reff_err = np.exp(bsrc[f'REFF_{rband}'][0])*conf.PIXEL_SCALE, np.exp(bsrc[f'REFF_{rband}'][0])*bsrc[f'REFF_ERR_{rband}'][0]*2.303*conf.PIXEL_SCALE
            ab, ab_err = bsrc[f'AB_{rband}'][0], bsrc[f'AB_ERR_{rband}'][0]
            if ab == -99.0:
                ab = -99
                ab_err = -99
            theta, theta_err = bsrc[f'THETA_{rband}'][0], bsrc[f'THETA_ERR_{rband}'][0]
        
            if 'Sersic' in src.name:
                nre, nre_err = bsrc[f'N_{rband}'][0], bsrc[f'N_ERR_{rband}'][0]

        # images
        img = blob.images[idx]
        wgt = blob.weights[idx]
        err = 1. / np.sqrt(wgt)
        mask = blob.masks[idx]
        seg = blob.segmap.copy()
        seg[blob.segmap != sid] = 0
        mod = blob.solution_model_images[idx]
        chi = blob.solution_tractor.getChiImage(idx)
        chi[blob.segmap != sid] = 0
        res = img - mod
        rms = np.median(blob.background_rms_images[idx])

        xpix, ypix = np.nonzero(seg)
        dx, dy = (np.max(xpix) - np.min(xpix)) / 2., (np.max(ypix) - np.min(ypix)) / 2.
        buff = np.min([conf.BLOB_BUFFER, 10.])
        xlim, ylim = np.array([-(dx + buff), (dx + buff)]) * conf.PIXEL_SCALE, np.array([-(dy + buff), (dy + buff)]) * conf.PIXEL_SCALE


        h, w = np.shape(img)
        dw, dh = w - xp - 1, h - yp - 1
        extent = np.array([-xp, dw, -yp, dh]) * conf.PIXEL_SCALE

        xp0, yp0 = (xp0 - xp) * conf.PIXEL_SCALE, (yp0 - yp) * conf.PIXEL_SCALE
        xp, yp = 0., 0.

        if is_resolved:
            aeff = reff #* conf.PIXEL_SCALE
            beff  = reff / ab #* conf.PIXEL_SCALE

            xa = xp + np.cos(np.deg2rad(90-theta)) * np.array([-1, 1]) * aeff
            ya = yp + np.sin(np.deg2rad(90-theta)) * np.array([-1, 1]) * aeff

            xb = xp + np.cos(np.deg2rad(theta)) * np.array([-1, 1]) * beff
            yb = yp + np.sin(np.deg2rad(theta)) * np.array([1, -1]) * beff


        # tests
        res_seg = res[blob.segmap==sid].flatten()
        try:
            k2, p_norm = stats.normaltest(res_seg)
        except:
            k2, p_norm = -99, -99
        chi_seg = chi[blob.segmap==sid].flatten()
        chi_sig = np.std(chi_seg)
        chi_mu = np.mean(chi_seg)

        # plotting
        fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(15, 15))

        # row 1 -- image, info
       
        if rms > 0.95*np.nanmax(img):
            normmin = 1.05*np.nanmin(abs(img))
        else:
            normmin = rms
        normmax = 0.95*np.nanmax(img)
        if normmin != normmax:
            norm = LogNorm(normmin, normmax, clip='True')
        else:
            norm=None
        ax[0,0].imshow(img, norm=norm, cmap='Greys', extent=extent)
        ax[0,0].text(0.05, 1.03, band, transform=ax[0,0].transAxes)
        ax[0,0].scatter(xp0, yp0, c='purple', marker='+', alpha=0.5)
        ax[0,0].scatter(xp, yp, c='royalblue', marker='x', alpha=0.9)
        ax[0,0].set(xlim=xlim, ylim=ylim)
        ax[0,1].axis('off')
        ax[0,2].axis('off')
        ax[0,3].axis('off')
        ax[0,1].text(0, 0.90,
                    s = f'Source: {sid} | Blob: {bid} | Brick: {blob.brick_id} | RA: {ra:6.6f}, Dec: {dec:6.6f}',
                    transform=ax[0,1].transAxes)
        if is_resolved:

            if 'Sersic' in src.name:
                ax[0,1].text(0, 0.70,
                    s = f'{src.name} with Reff: {reff:3.3f}+/-{reff_err:3.3f}, n: {nre:3.3f}+/-{nre_err:3.3f}, A/B: {ab:3.3f}+/-{ab_err:3.3f}, Theta: {theta:3.3f}+/-{theta_err:3.3f}',
                    transform=ax[0,1].transAxes)
            else:
                ax[0,1].text(0, 0.70,
                    s = f'{src.name} with Reff: {reff:3.3f}+/-{reff_err:3.3f}, A/B: {ab:3.3f}+/-{ab_err:3.3f}, and Theta: {theta:3.3f}+/-{theta_err:3.3f}',
                    transform=ax[0,1].transAxes)
        else:
            ax[0,1].text(0, 0.70,
                    s = f'{src.name}',
                    transform=ax[0,1].transAxes)
        ax[0,1].text(0, 0.50,
                    s = f'{band} | {flux:3.3f}+/-{flux_err:3.3f} uJy | {mag:3.3f}+/-{mag_err:3.3f} AB | S/N = {snr:3.3f}',
                    transform=ax[0,1].transAxes)
        ax[0,1].text(0, 0.30,
                    s = f'Chi2/N: {chi2:3.3f} | N_blob: {n_blob} | '+r'$\mu(\chi)$'+f'={chi_mu:3.3f}, '+r'$\sigma(\chi)$'+f'={chi_sig:3.3f} | K2-test: {k2:3.3f}',
                    transform=ax[0,1].transAxes)
        

        # row 2 -- image, weights, mask, segment
        ax[1,0].imshow(img, vmin=-3*rms, vmax=3*rms, cmap='RdGy', extent=extent)
        ax[1,0].text(0.05, 1.03, 'Image', transform=ax[1,0].transAxes)
        ax[1,0].set(xlim=xlim, ylim=ylim)
        ax[1,0].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        ax[1,1].imshow(err, cmap='Greys', extent=extent)
        ax[1,1].text(0.05, 1.03, r'med($\sigma$)'+f'={rms*10**(-0.4 * (zpt - 23.9)):5.5f} uJy', transform=ax[1,1].transAxes)
        ax[1,1].set(xlim=xlim, ylim=ylim)
        ax[1,1].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        ax[1,2].imshow(mask, cmap='Greys', extent=extent)
        ax[1,2].text(0.05, 1.03, 'Blob', transform=ax[1,2].transAxes)
        ax[1,2].set(xlim=xlim, ylim=ylim)
        ax[1,2].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        ax[1,3].imshow(~seg, cmap='Greys', extent=extent)
        ax[1,3].text(0.05, 1.03, 'Segment', transform=ax[1,3].transAxes)
        ax[1,3].set(xlim=xlim, ylim=ylim)
        ax[1,3].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)

        ax[1,0].scatter(xp, yp, c='royalblue', marker='x')
        ax[1,2].scatter(xp0, yp0, c='purple', marker='+', alpha=0.5)
        ax[1,2].scatter(xp, yp, c='royalblue', marker='x', alpha=0.9)
        ax[1,3].scatter(xp0, yp0, c='purple', marker='+', alpha=0.5)
        ax[1,3].scatter(xp, yp, c='royalblue', marker='x', alpha=0.9)

        ax[1,2].plot()

        # row 3 -- image, model, residual, chi
        ax[2,0].imshow(img/err, vmin=-3, vmax=3, cmap='RdGy', extent=extent)
        ax[2,0].text(0.05, 1.03, 'S/N', transform=ax[2,0].transAxes)
        ax[2,0].set(xlim=xlim, ylim=ylim)
        ax[2,0].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        # ax[2,1].imshow(mod, vmin=-3*rms, vmax=3*rms, cmap='RdGy', extent=extent)
        ax[2,1].imshow(mod, norm=norm, cmap='Greys', extent=extent)
        ax[2,1].text(0.05, 1.03, 'Model', transform=ax[2,1].transAxes)
        ax[2,1].set(xlim=xlim, ylim=ylim)
        ax[2,1].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        ax[2,2].imshow(res, vmin=-3*rms, vmax=3*rms, cmap='RdGy', extent=extent)
        ax[2,2].text(0.05, 1.03, 'Residual', transform=ax[2,2].transAxes)
        ax[2,2].set(xlim=xlim, ylim=ylim)
        ax[2,2].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)
        ax[2,3].imshow(chi, vmin=-3, vmax=3, cmap='RdGy', extent=extent)
        ax[2,3].text(0.05, 1.03, r'$\chi$', transform=ax[2,3].transAxes)
        ax[2,3].set(xlim=xlim, ylim=ylim)
        ax[2,3].contour(img, levels=np.arange(2*rms, np.min([5*rms, np.max(img)]), rms), colors='royalblue', extent=extent, alpha=0.5)

        if is_resolved:
            ax[2,0].plot(xa, ya, c='royalblue', alpha=0.7)
            ax[2,0].plot(xb, yb, c='royalblue', alpha=0.7)
            ax[2,1].plot(xa, ya, c='royalblue', alpha=0.7)
            ax[2,1].plot(xb, yb, c='royalblue', alpha=0.7)
            ax[2,2].plot(xa, ya, c='royalblue', alpha=0.7)
            ax[2,2].plot(xb, yb, c='royalblue', alpha=0.7)
            ax[2,3].plot(xa, ya, c='royalblue', alpha=0.7)
            ax[2,3].plot(xb, yb, c='royalblue', alpha=0.7)
        else:
            ax[2,0].scatter(xp, yp, c='royalblue', marker='x', alpha=0.7)
            ax[2,1].scatter(xp, yp, c='royalblue', marker='x', alpha=0.7)
            ax[2,2].scatter(xp, yp, c='royalblue', marker='x', alpha=0.7)
            ax[2,3].scatter(xp, yp, c='royalblue', marker='x', alpha=0.7)

        # row 4 -- psf, x-slice, y-slice, hist

        psfmodel = blob.psfimg[band]
        xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2 + 0.5)
        [ax[3,0].plot(xax * 0.15, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[0])]
        ax[3,0].axvline(0, ls='dotted', c='k')
        ax[3,0].set(xlim=(-5, 5), yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')
        ax[3,0].text(0.05, 1.03, 'PSF', transform=ax[3,0].transAxes)

        # x slice
        imgx = blob.images[idx][:, int(xps)]
        errx = 1./np.sqrt(blob.weights[idx][:, int(xps)])
        modx = blob.solution_model_images[idx][:, int(xps)]
        sign = 1
        if bsrc[f'RAWFLUX_{band}'][0] < 0:
            sign = -1
        modxlo = blob.solution_model_images[idx][:, int(xps)] / bsrc[f'RAWFLUX_{band}'][0] * (bsrc[f'RAWFLUX_{band}'][0] - sign * bsrc[f'RAWFLUXERR_{band}'][0])
        modxhi = blob.solution_model_images[idx][:, int(xps)] / bsrc[f'RAWFLUX_{band}'][0] * (bsrc[f'RAWFLUX_{band}'][0] + sign * bsrc[f'RAWFLUXERR_{band}'][0])
        resx = imgx - modx

        # y slice
        imgy = blob.images[idx][int(yps), :]
        erry = 1./np.sqrt(blob.weights[idx][int(yps), :])
        mody = blob.solution_model_images[idx][int(yps), :]
        if bsrc[f'RAWFLUX_{band}'][0] < 0:
            sign = -1
        modylo = blob.solution_model_images[idx][int(yps), :] / bsrc[f'RAWFLUX_{band}'][0] * (bsrc[f'RAWFLUX_{band}'][0] - sign * bsrc[f'RAWFLUXERR_{band}'][0])
        modyhi = blob.solution_model_images[idx][int(yps), :] / bsrc[f'RAWFLUX_{band}'][0] * (bsrc[f'RAWFLUX_{band}'][0] + sign * bsrc[f'RAWFLUXERR_{band}'][0])
        resy = imgy - mody

        ylim = (0.9*np.min([np.min(imgx), np.min(imgy)]), 1.1*np.max([np.max(imgx), np.max(imgy)]))

        xax = np.linspace(extent[2], extent[3]+conf.PIXEL_SCALE, len(imgx))
        ax[3,2].errorbar(xax, imgx, yerr=errx, c='k')
        ax[3,2].plot(xax, modx, c='r')
        ax[3,2].fill_between(xax, modxlo, modxhi, color='r', alpha=0.3)
        ax[3,2].plot(xax, resx, c='g')
        ax[3,2].axvline(0, ls='dotted', c='k')
        ax[3,2].set(ylim =ylim,  xlabel='arcsec', xlim=xlim)
        ax[3,2].text(0.05, 1.03, 'Y', transform=ax[3,2].transAxes)

        yax = np.linspace(extent[0], extent[1]+conf.PIXEL_SCALE, len(imgy))
        ax[3,1].errorbar(yax, imgy, yerr=erry, c='k')
        ax[3,1].plot(yax, mody, c='r')
        ax[3,1].fill_between(yax, modylo, modyhi, color='r', alpha=0.3)
        ax[3,1].plot(yax, resy, c='g')
        ax[3,1].axvline(0, ls='dotted', c='k')
        ax[3,1].set(ylim=ylim, xlabel='arcsec', xlim=xlim)
        ax[3,1].text(0.05, 1.03, 'X', transform=ax[3,1].transAxes)

        hist(chi_seg, ax=ax[3,3], bins='freedman', histtype='step', density=True)
        ax[3,3].axvline(0, ls='dotted', color='grey')
        ax[3,3].text(0.05, 1.03, 'Residual '+r'$\sigma(\chi)$'+f'={chi_sig:3.3f}', transform=ax[3,3].transAxes)
        ax[3,3].set(xlim=(-10, 10), xlabel=r'$\chi$')
        ax[3,3].axvline(chi_mu, c='royalblue', ls='dashed')
        ax[3,3].axvline(0, c='grey', ls='dashed', alpha=0.3)
        ax[3,3].axvline(chi_mu-chi_sig, c='royalblue', ls='dotted')
        ax[3,3].axvline(chi_mu+chi_sig, c='royalblue', ls='dotted')
        ax[3,3].axvline(-1, c='grey', ls='dotted', alpha=0.3)
        ax[3,3].axvline(1, c='grey', ls='dotted', alpha=0.3)


        pdf.savefig(fig)
        plt.close()
    logger.info(f'Saving figure: {outpath}') 
    pdf.close()

def plot_apertures(blob, band=None):
    pass

def plot_iterblob(blob, tr, iteration, bands=None):

    if blob._is_itemblob:
        sid = blob.bcatalog['source_id'][0]
        if bands is None:
            band = conf.MODELING_NICKNAME
            nickname = conf.MODELING_NICKNAME
            bidx = [0,]
            bands = [band,]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')
        else:
            bidx = [blob._band2idx(b, bands=blob.bands) for b in bands]
            if bands[0].startswith(conf.MODELING_NICKNAME):
                nickname = conf.MODELING_NICKNAME
            else:
                nickname = conf.MULTIBAND_NICKNAME
            if len(bands) > 1:
                outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{nickname}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')
            else:
                outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{bands[0]}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')

    else:
        if bands is None:
            band = conf.MODELING_NICKNAME
            nickname = conf.MODELING_NICKNAME
            bidx = [0,]
            bands = [band,]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{conf.MODELING_NICKNAME}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')
        else:
            bidx = [blob._band2idx(b, bands=blob.bands) for b in bands]
            if bands[0].startswith(conf.MODELING_NICKNAME):
                nickname = conf.MODELING_NICKNAME
            else:
                nickname = conf.MULTIBAND_NICKNAME
            if len(bands) > 1:
                outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{nickname}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')
            else:
                outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{bands[0]}_{blob._level}_{blob._sublevel}_{iteration}_iterblob.pdf')

    import matplotlib.backends.backend_pdf
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath)
    
    for idx, band in zip(bidx, bands):
        
        if band == conf.MODELING_NICKNAME:
            zpt = conf.MODELING_ZPT
        elif band.startswith(conf.MODELING_NICKNAME):
            band_name = band[len(conf.MODELING_NICKNAME)+1:]
            zpt = conf.MULTIBAND_ZPT[blob._band2idx(band_name)]
        else:
            zpt = conf.MULTIBAND_ZPT[blob._band2idx(band)]

        cat = tr.getCatalog()
        xp, yp = [src.pos[0] for src in cat], [src.pos[1] for src in cat]

        back = blob.background_images[idx]
        back_rms = blob.background_rms_images[idx]
        mean, rms =  np.nanmean(back), np.nanmean(back_rms)
        
        img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

        # image
        img = blob.images[idx]
        # model
        mod = tr.getModelImage(idx)
        # residual
        res = img - mod
        # chi2
        chi2 = tr.getChiImage(idx)

        fig, ax = plt.subplots(ncols=4)
        ax[0].imshow(img, **img_opt)
        ax[1].imshow(mod, **img_opt)
        ax[2].imshow(res, **img_opt)
        ax[3].imshow(chi2, cmap='RdGy', vmin=-5, vmax=5)

        fig.suptitle(f'Blob {blob.blob_id} | {band} | iter: {iteration}')

        [ax[i].scatter(xp, yp, marker='x', c='royalblue') for i in np.arange(4)]
        [ax[i].set_title(title, fontsize=20) for i, title in enumerate(('Image', 'Model', 'Image-Model', '$\chi^{2}$'))]

        pdf.savefig(fig)
        plt.close()
    logger.info(f'Saving figure: {outpath}') 
    pdf.close()

def plot_modprofile(blob, band=None):

    if band is None:
        band = conf.MODELING_NICKNAME
        idx = 0
    else:
        idx = blob._band2idx(band, bands=blob.bands)

    psfmodel = blob.psfimg[band]

    back = blob.background_images[idx]
    back_rms = blob.background_rms_images[idx]
    mean, rms =  np.nanmean(back), np.nanmean(back_rms)
    noise = np.random.normal(mean, rms, size=blob.dims)
    tr = blob.solution_tractor
    
    norm = LogNorm(mean + 3*rms, blob.images[idx].max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)
    img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

    xlim = (-np.shape(blob.images[idx])[1]/2,  np.shape(blob.images[idx])[1]/2)

    fig, ax = plt.subplots(ncols = 5, nrows = 2, figsize=(20,10))
    ax[1,0].imshow(blob.images[idx], **img_opt)
    ax[1,1].imshow(blob.solution_model_images[idx],  **img_opt)
    residual = blob.images[idx] - blob.solution_model_images[idx]
    ax[1,2].imshow(residual, **img_opt)

    xax = np.arange(-np.shape(blob.images[idx])[1]/2,  np.shape(blob.images[idx])[1]/2)
    [ax[0,0].plot(xax * 0.15, blob.images[idx][x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(blob.images[idx])[0])]
    ax[0,0].axvline(0, ls='dotted', c='k')
    ax[0,0].set(yscale='log', xlabel='arcsec')

    xax = np.arange(-np.shape(blob.solution_model_images[idx])[1]/2,  np.shape(blob.solution_model_images[idx])[1]/2)
    [ax[0,1].plot(xax * 0.15, blob.solution_model_images[idx][x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(blob.solution_model_images[idx])[0])]
    ax[0,1].axvline(0, ls='dotted', c='k')
    ax[0,1].set(yscale='log', xlabel='arcsec')

    xax = np.arange(-np.shape(residual)[1]/2,  np.shape(residual)[1]/2)
    [ax[0,2].plot(xax * 0.15, residual[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(residual)[0])]
    ax[0,2].axvline(0, ls='dotted', c='k')
    ax[0,2].set(yscale='log', xlabel='arcsec')

    norm = LogNorm(1e-5, 0.1*np.nanmax(psfmodel), clip='True')
    img_opt = dict(cmap='Blues', norm=norm)
    ax[1,3].imshow(psfmodel, norm=norm, extent=0.15 *np.array([-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2, -np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2,]))
    ax[1,3].set(xlim=xlim, ylim=xlim)

    xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2 + 0.5)
    [ax[0,3].plot(xax * 0.15, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[0])]
    ax[0,3].axvline(0, ls='dotted', c='k')
    ax[0,3].set(xlim=xlim, yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')

    for j, src in enumerate(blob.solution_catalog):
        try:
            mtype = src.name
        except:
            mtype = 'PointSource'
        flux = src.getBrightness().getFlux(band)
        chisq = blob.solution_chisq[j, idx]
        band = band.replace(' ', '_')
        if band == conf.MODELING_NICKNAME:
            zpt = conf.MODELING_ZPT
        elif band.startswith(conf.MODELING_NICKNAME):
            band_name = band[len(conf.MODELING_NICKNAME)+1:]
            zpt = conf.MULTIBAND_ZPT[blob._band2idx(band_name)]
        else:
            zpt = conf.MULTIBAND_ZPT[blob._band2idx(band)]
            
        mag = zpt - 2.5 * np.log10(flux)

        topt = dict(color=colors[j], transform = ax[0, 3].transAxes)
        ystart = 0.99 - j * 0.5
        ax[0, 4].text(1.05, ystart - 0.1, f'{j}) {mtype}', **topt)
        ax[0, 4].text(1.05, ystart - 0.2, f'  F({band}) = {flux:4.4f}', **topt)
        ax[0, 4].text(1.05, ystart - 0.3, f'  M({band}) = {mag:4.4f}', **topt)
        ax[0, 4].text(1.05, ystart - 0.4, f'  zpt({band}) = {zpt:4.4f}', **topt)
        ax[0, 4].text(1.05, ystart - 0.5, f'  $\chi^{2}$ = {chisq:4.4f}', **topt)
    
    ax[0, 4].axis('off')
    ax[1, 4].axis('off')

    for i in np.arange(3):
        ax[0, i].set(xlim=(0.15*xlim[0], 0.15*xlim[1]), ylim=(np.nanmedian(blob.images[idx]), blob.images[idx].max()))
        # ax[1, i].set(xlim=(-15, 15), ylim=(-15, 15))
    ax[0, 3].set(xlim=(0.15*xlim[0], 0.15*xlim[1]))
    if blob._is_itemblob:
            sid = blob.bcatalog['source_id'][0]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{band}_debugprofile.pdf')
    else:
        outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{band}_debugprofile.pdf')
    logger.info(f'Saving figure: {outpath}') 
    fig.savefig(outpath)
    plt.close()

def plot_xsection(blob, band, src, sid):
    if band is None:
        band = conf.MODELING_NICKNAME
        idx = 0
    else:
        idx = blob._band2idx(band, bands=blob.bands)

    back = blob.background_images[idx]
    back_rms = blob.background_rms_images[idx]
    mean, rms =  np.nanmean(back), np.nanmean(back_rms)

    fig, ax = plt.subplots(ncols=2)

    posx, posy = src.pos[0], src.pos[1]
    try:

        # x slice
        imgx = blob.images[idx][:, int(posx)]
        errx = 1/np.sqrt(blob.weights[idx][:, int(posx)])
        modx = blob.solution_model_images[idx][:, int(posx)]
        resx = imgx - modx

        # y slice
        imgy = blob.images[idx][int(posy), :]
        erry = 1/np.sqrt(blob.weights[idx][int(posy), :])
        mody = blob.solution_model_images[idx][int(posy), :]
        resy = imgy - mody
    
    except:
        plt.close()
        logger.warning('Could not make plot -- object may have escaped?')
        return

    # idea: show areas outside segment in grey

    ylim = (0.9*np.min([np.min(imgx), np.min(imgy)]), 1.1*np.max([np.max(imgx), np.max(imgy)]))

    xax = np.arange(-np.shape(blob.images[idx])[0]/2,  np.shape(blob.images[idx])[0]/2) * conf.PIXEL_SCALE
    ax[0].errorbar(xax, imgx, yerr=errx, c='k')
    ax[0].plot(xax, modx, c='r')
    ax[0].plot(xax, resx, c='g')
    ax[0].axvline(0, ls='dotted', c='k')
    ax[0].set(ylim =ylim,  xlabel='arcsec')

    yax = np.arange(-np.shape(blob.images[idx])[1]/2,  np.shape(blob.images[idx])[1]/2) * conf.PIXEL_SCALE
    ax[1].errorbar(yax, imgy, yerr=erry, c='k')
    ax[1].plot(yax, mody, c='r')
    ax[1].plot(yax, resy, c='g')
    ax[1].axvline(0, ls='dotted', c='k')
    ax[1].set(ylim=ylim, xlabel='arcsec')

    # for j, src in enumerate(blob.solution_catalog):
    #     try:
    #         mtype = src.name
    #     except:
    #         mtype = 'PointSource'
    #     flux = src.getBrightness().getFlux(band)
    #     chisq = blob.solution_chisq[j, idx]
    #     band = band.replace(' ', '_')
    #     if band == conf.MODELING_NICKNAME:
    #         zpt = conf.MODELING_ZPT
    #     else:
    #         zpt = conf.MULTIBAND_ZPT[idx]
    #     mag = zpt - 2.5 * np.log10(flux)

    #     topt = dict(color=colors[j], transform = ax[0, 3].transAxes)
    #     ystart = 0.99 - j * 0.4
    #     ax[0, 4].text(1.05, ystart - 0.1, f'{j}) {mtype}', **topt)
    #     ax[0, 4].text(1.05, ystart - 0.2, f'  F({band}) = {flux:4.4f}', **topt)
    #     ax[0, 4].text(1.05, ystart - 0.3, f'  M({band}) = {mag:4.4f}', **topt)
    #     ax[0, 4].text(1.05, ystart - 0.4, f'  $\chi^{2}$ = {chisq:4.4f}', **topt)
    
    outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{band}_xsection.pdf')
    logger.info(f'Saving figure: {outpath}') 
    fig.savefig(outpath)
    plt.close()

def plot_detblob(blob, fig=None, ax=None, band=None, level=0, sublevel=0, final_opt=False, init=False):

    if band is None:
        idx = 0
        band = ''
    else:
        # print(blob.bands)
        # print(band)
        idx = np.argwhere(np.array(blob.bands) == band)[0][0]

    back = blob.background_images[idx]
    rms = blob.background_rms_images[idx]
    mean, rms = np.nanmean(back), np.nanmean(rms)
    noise = np.random.normal(mean, rms, size=blob.dims)
    tr = blob.solution_tractor

    norm = LogNorm(np.max([mean + rms, 1E-5]), blob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)
    img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

    # Init
    if fig is None:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(24,48), ncols=6, nrows=13)

        # Detection image
        ax[0,0].imshow(blob.images[idx], **img_opt)
        [ax[0,i].axis('off') for i in np.arange(1, 6)]

        if blob.n_sources == 1:
            objects = blob.bcatalog
            e = Ellipse(xy=(objects['x'], objects['y']),
                        width=6*objects['a'],
                        height=6*objects['b'],
                        angle=objects['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor(colors[0])
            ax[0, 0].add_artist(e)
        else:
            for j, src in enumerate(blob.bcatalog):
                objects = blob.bcatalog[j]
                e = Ellipse(xy=(objects['x'], objects['y']),
                            width=6*objects['a'],
                            height=6*objects['b'],
                            angle=objects['theta'] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor(colors[j])
                ax[0, 0].add_artist(e)

        ax[0,1].text(0.1, 0.9, f'Blob #{blob.blob_id}', transform=ax[0,1].transAxes)
        ax[0,1].text(0.1, 0.8, f'{blob.n_sources} source(s)', transform=ax[0,1].transAxes)

        [ax[1,i+1].set_title(title, fontsize=20) for i, title in enumerate(('Model', 'Model+Noise', 'Image-Model', '$\chi^{2}$', 'Residuals'))]

        if blob._is_itemblob:
            sid = blob.bcatalog['source_id'][0]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_{band}.pdf')
        else:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{conf.MODELING_NICKNAME}_{band}.pdf')
        fig.savefig(outpath)
        logger.info(f'Saving figure: {outpath}')

    elif final_opt:
        nrow = 4 * level + 2* sublevel + 2
        [[ax[i,j].axis('off') for i in np.arange(nrow+1, 11)] for j in np.arange(0, 6)]

        ax[11,0].axis('off')
        residual = blob.images[idx] - blob.pre_solution_model_images[idx]
        ax[11,1].imshow(blob.pre_solution_model_images[idx], **img_opt)
        ax[11,2].imshow(blob.pre_solution_model_images[idx] + noise, **img_opt)
        ax[11,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[11,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[11,5].hist(res_seg, bins=20, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
        ax[11,5].set_xlim(minx, maxx)
        ax[11,5].axvline(0, c='grey', ls='dotted')
        ax[11,5].set_ylim(bottom=0)

        ax[12,0].axis('off')
        residual = blob.images[idx] - blob.solution_model_images[idx]
        ax[12,1].imshow(blob.solution_model_images[idx], **img_opt)
        ax[12,2].imshow(blob.solution_model_images[idx] + noise, **img_opt)
        ax[12,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[12,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        ax[12,1].set_ylabel('Solution')

        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[12,5].hist(res_seg, bins=20, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
        ax[12,5].set_xlim(minx, maxx)
        ax[12,5].axvline(0, c='grey', ls='dotted')
        ax[12,5].set_ylim(bottom=0)

        # Solution params
        for i, src in enumerate(blob.solution_catalog):
            ax[1,0].text(0.1, 0.8 - 0.4*i, f'#{blob.bcatalog[i]["source_id"]} Model:{src.name}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.1, f'       Flux: {src.brightness[0]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.2, '       '+r'$\chi^{2}$/N:'+f'{blob.solution_chisq[i,0]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)


        #fig.subplots_adjust(wspace=0, hspace=0)
        if blob._is_itemblob:
            sid = blob.bcatalog['source_id'][0]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_{band}.pdf')
        else:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{conf.MODELING_NICKNAME}_{band}.pdf')
        fig.savefig(outpath)
        plt.close()
        logger.info(f'Saving figure: {outpath}')

    else:
        if conf.DECISION_TREE == 1:
            if init:
                nrow = 4 * level + 2 * sublevel + 1
            else:
                nrow = 4 * level + 2 * sublevel + 2
        elif conf.DECISION_TREE == 2:
            if level == 0:
                if sublevel == 0:
                    nrow = 1
                if sublevel == 1:
                    nrow = 3
            if level == 1:
                if sublevel == 0:
                    nrow = 5
            if level == 2:
                if sublevel == 0:
                    nrow = 7
            if level == 3:
                if sublevel == 0:
                    nrow = 9
            if not init:
                nrow += 1
        
        residual = blob.images[idx] - blob.tr.getModelImage(idx)
        ax[nrow,0].axis('off')
        ax[nrow,1].imshow(blob.tr.getModelImage(idx), **img_opt)
        ax[nrow,2].imshow(blob.tr.getModelImage(idx) + noise, **img_opt)
        ax[nrow,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[nrow,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        if conf.DECISION_TREE == 1:
            models = {1:'PointSource', 3:'SimpleGalaxy', 5:'ExpGalaxy', 7:'DevGalaxy', 9:'CompositeGalaxy'}
        elif conf.DECISION_TREE == 2:
            models = {1:'PointSource', 3:'SimpleGalaxy', 5:'SersicGalaxy', 7:'SersicCoreGalaxy', 9:'CompositeGalaxy'}

        if init:
            ax[nrow,1].set_ylabel(models[nrow])
        
        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            
            if np.shape(residual) != np.shape(blob.segmap):
                plt.figure()
                plt.imshow(blob.segmap, cmap='Greys', norm=LogNorm())
                plt.savefig(os.path.join(conf.PLOT_DIR,'debug_segmap.pdf'))
                plt.figure()
                plt.imshow(residual, cmap='Greys', norm=LogNorm())
                plt.savefig(os.path.join(conf.PLOT_DIR,'debug_residual.pdf'))
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[nrow,5].hist(res_seg, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
            if not init:
                ax[nrow,4].text(0.02, 0.9 - 0.1*i, r'$\chi^{2}$/N'+f'={blob.rchisq[i, level, sublevel]:2.2f} | BIC={blob.bic[i, level, sublevel]:2.2f}',
                         color=colors[i], transform=ax[nrow,4].transAxes)
        ax[nrow,5].set_xlim(minx, maxx)
        ax[nrow,5].axvline(0, c='grey', ls='dotted')
        ax[nrow,5].set_ylim(bottom=0)

        for i, src in enumerate(blob.bcatalog):
            x, y = src['x'], src['y']
            color = colors[i]
            if not blob._solved[i]:
                ax[nrow,1].plot([x, x], [y - 10, y - 5], ls='dotted', c=color)
                ax[nrow,1].plot([x - 10, x - 5], [y, y], ls='dotted', c=color)
            else:
                ax[nrow,1].plot([x, x], [y - 10, y - 5], c=color)
                ax[nrow,1].plot([x - 10, x - 5], [y, y], c=color)

        if blob._is_itemblob:
            sid = blob.bcatalog['source_id'][0]
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_S{sid}_{conf.MODELING_NICKNAME}_{band}.pdf')
        else:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{conf.MODELING_NICKNAME}_{band}.pdf')
        fig.savefig(outpath)
        logger.info(f'Saving figure: {outpath}')

    return fig, ax

def plot_fblob(blob, band, fig=None, ax=None, final_opt=False, debug=False):

    idx = np.argwhere(blob.bands == band)[0][0]
    back = blob.backgrounds[idx]
    mean, rms = back[0], back[1]
    noise = np.random.normal(mean, rms, size=blob.dims)
    tr = blob.solution_tractor
    
    norm = LogNorm(np.max([mean + rms, 1E-5]), 0.98*blob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)
    img_opt = dict(cmap='RdGy', vmin=-5*rms, vmax=5*rms)

    if final_opt:
        
        ax[2,0].axis('off')
        residual = blob.images[idx] - blob.solution_model_images[idx]
        ax[2,1].imshow(blob.solution_model_images[idx], **img_opt)
        ax[2,2].imshow(blob.solution_model_images[idx] + noise, **img_opt)
        ax[2,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[2,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        ax[2,1].set_ylabel('Solution')

        bins = np.arange(np.nanpercentile(residua, 5), np.nanpercentile(residual, 95), 0.1)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            img_seg = blob.images[idx][blob.segmap==src['source_id']].flatten()
            ax[2,5].hist(img_seg, bins=bins, linestyle='dotted', histtype='step', color=colors[i], density=True)
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[2,5].hist(res_seg, bins=bins, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanpercentile(res_seg, 5), np.nnanpercentile(res_seg, 95)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
        ax[2,5].set_xlim(-5, 5) #minx, maxx)
        ax[2,5].axvline(0, c='grey', ls='dotted')
        ax[2,5].set_ylim(bottom=0)

        dof = '/N'

        # Solution params
        for i, src in enumerate(blob.solution_catalog):
            original_zpt = np.array(conf.MULTIBAND_ZPT)[idx]
            target_zpt = 23.9
            flux_ujy = src.getBrightness().getFlux(band) * 10 ** (0.4 * (target_zpt - original_zpt))
            flux_var = blob.forced_variance
            fluxerr_ujy = np.sqrt(flux_var[i].brightness.getParams()[idx]) * 10**(0.4 * (target_zpt - original_zpt))
            ax[1,0].text(0.1, 0.8 - 0.4*i, f'#{blob.bcatalog[i]["source_id"]} Model:{src.name}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.1, f'       Flux: {flux_ujy:3.3f}+\-{fluxerr_ujy:3.3f} uJy', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.2, f'       Chi2{dof}: {blob.solution_chisq[i,idx]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)

        #fig.subplots_adjust(wspace=0, hspace=0)
        outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{blob.bands[idx]}.pdf')
        fig.savefig(outpath)
        plt.close()
        logger.info(f'Saving figure: {outpath}')

    else:
        # Init
        if fig is None:
            plt.ioff()
            fig, ax = plt.subplots(figsize=(24,8), ncols=6, nrows=3)

            # Detection image
            ax[0,0].imshow(blob.images[idx], **img_opt)
            [ax[0,i].axis('off') for i in np.arange(1, 6)]

            for j, src in enumerate(blob.bcatalog):
                objects = blob.bcatalog[j]
                position = [src[f'x'], src[f'y']]

                position[0] -= (blob.subvector[1] + blob.mosaic_origin[1] - conf.BRICK_BUFFER)
                position[1] -= (blob.subvector[0] + blob.mosaic_origin[0] - conf.BRICK_BUFFER)
                e = Ellipse(xy=(position[0], position[1]),
                            width=6*objects['a'],
                            height=6*objects['b'],
                            angle=objects['theta'] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor(colors[j])
                ax[0, 0].add_artist(e)

        ax[0,1].text(0.1, 0.9, f'{band} | Blob #{blob.blob_id}', transform=ax[0,1].transAxes)
        ax[0,1].text(0.1, 0.8, f'{blob.n_sources} source(s)', transform=ax[0,1].transAxes)

        [ax[0,j].axis('off') for j in np.arange(1, 6)]

        ax[1,0].axis('off')
        residual = blob.images[idx] - blob.tr.getModelImage(idx)
        ax[1,0].axis('off')
        ax[1,1].imshow(blob.tr.getModelImage(idx), **img_opt)
        ax[1,2].imshow(blob.tr.getModelImage(idx) + noise, **img_opt)
        ax[1,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[1,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            try:
                k2, p_norm = stats.normaltest(res_seg)
            except:
                k2, p_norm = -99, -99
            ax[1,5].hist(res_seg, bins=20, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
            ax[1,5].text(0.05, 0.9 - 0.1*i, s=f"k={k2:3.3f} (p={p_norm:2.2f})", trasnform=ax[1,5].transAxes)
        ax[1,5].set_xlim(minx, maxx)
        ax[1,5].axvline(0, c='grey', ls='dotted')
        ax[1,5].set_ylim(bottom=0)

        [ax[1,i+1].set_title(title, fontsize=20) for i, title in enumerate(('Model', 'Model+Noise', 'Image-Model', '$\chi^{2}$', 'Residuals'))]

        if debug:
            outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{blob.bands[idx]}_DEBUG.pdf')
            fig.savefig(outpath)
            plt.close()
            logger.info(f'DEBUG | Saving figure: {outpath}')
    
    return fig, ax

def plot_blobmap(brick, image=None, band=None, catalog=None, mode='rms'):
    if image is None:
        image = brick.images[0]
    if band is None:
        band = brick.bands[0]
    if catalog is None:
        catalog = brick.catalog
    fig, ax = plt.subplots(figsize=(20,20))
    # imgs_marked = mark_boundaries(brick.images[0], brick.blobmap, color='red')[:,:,0]
    imgs_marked = find_boundaries(brick.blobmap, mode='thick').astype(int)
    imgs_marked[imgs_marked==0] = -99
    backlevel, noisesigma = brick.backgrounds[0]
    if mode == 'log':
        vmin, vmax = np.max([backlevel + noisesigma, 1E-5]), brick.images[0].max()
        norm = LogNorm(np.max([backlevel + noisesigma, 1E-5]), 0.9*np.max(image), clip='True')
        ax.imshow(image, cmap='Greys', origin='lower', norm=norm)
    elif mode == 'rms':
        ax.imshow(image - backlevel, cmap='RdGy', origin='lower', vmin=-3*noisesigma, vmax=3*noisesigma)
    mycmap = plt.cm.Greens
    mycmap.set_under('k', alpha=0)
    ax.imshow(imgs_marked, alpha=0.9, cmap=mycmap, vmin=0, zorder=2, origin='lower')
    ax.scatter(catalog['x'], catalog['y'], marker='+', color='limegreen', s=0.1)
    ax.add_patch(Rectangle((conf.BRICK_BUFFER, conf.BRICK_BUFFER), conf.BRICK_HEIGHT, conf.BRICK_WIDTH, fill=False, alpha=0.3, edgecolor='purple', linewidth=1))

    for i in np.arange(brick.n_blobs):
        idx, idy = np.nonzero(brick.blobmap == i+1)
        xlo, xhi = np.min(idx) - conf.BLOB_BUFFER, np.max(idx) + 1 + conf.BLOB_BUFFER
        ylo, yhi = np.min(idy) - conf.BLOB_BUFFER, np.max(idy) + 1 + conf.BLOB_BUFFER
        w = xhi - xlo #+ 2 * conf.BLOB_BUFFER
        h = yhi - ylo #+ 2 * conf.BLOB_BUFFER
        rect = Rectangle((ylo, xlo), h, w, fill=False, alpha=0.3,
                                edgecolor='red', zorder=3, linewidth=1)
        ax.add_patch(rect)
        ax.annotate(str(i+1), (ylo, xlo), color='r', fontsize=2)
        #ax.scatter(x + width/2., y + height/2., marker='+', c='r')

    # Add collection to axes
    #ax.axis('off')
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_{mode}_blobmaster.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.suptitle(brick.bands[0])
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    logger.info(f'Saving figure: {out_path}')

def plot_ldac(tab_ldac, band, xlims=None, ylims=None, box=False, sel=None):
    fig, ax = plt.subplots()
    xbin = np.arange(0, 15, 0.1)
    ybin = np.arange(12, 26, 0.1)
    ax.hist2d(tab_ldac['FLUX_RADIUS'], tab_ldac['MAG_AUTO'], bins=(xbin, ybin), cmap='Greys', norm=LogNorm())
    if box:
        rect = Rectangle((xlims[0], ylims[0]), xlims[1] - xlims[0], ylims[1] - ylims[0], fill=False, alpha=0.3,
                                edgecolor='r', facecolor=None, zorder=3, linewidth=1)
        ax.add_patch(rect)
    if (sel is not None) & box:
        ax.scatter(tab_ldac['FLUX_RADIUS'][sel], tab_ldac['MAG_AUTO'][sel], s=0.1, c='r')

    fig.subplots_adjust(bottom = 0.15)
    ax.set(xlabel='Flux Radius (px)', xlim=(0, 15),
            ylabel='Mag Auto (AB)', ylim=(26, 12))
    ax.grid()

    if sel is not None:
        nsel = np.sum(sel)
        ax.text(x=0.05, y=0.95, s=f'N = {nsel}', transform=ax.transAxes)
    fig.savefig(os.path.join(conf.PLOT_DIR, f'{band}_box_{box}_ldac.pdf'), overwrite=True)

    plt.close()

def plot_psf(psfmodel, band, show_gaussian=False):

    fig, ax = plt.subplots(ncols=3, figsize=(30,10))
    norm = LogNorm(1e-8, 0.1*np.nanmax(psfmodel), clip='True')
    img_opt = dict(cmap='Blues', norm=norm)
    ax[0].imshow(psfmodel, **img_opt, extent=conf.PIXEL_SCALE *np.array([-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2, -np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2,]))
    ax[0].set(xlim=(-15,15), ylim=(-15, 15))
    ax[0].axvline(0, color='w', ls='dotted')
    ax[0].axhline(0, color='w', ls='dotted')

    xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2+0.5)
    [ax[1].plot(xax * conf.PIXEL_SCALE, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[1])]
    ax[1].axvline(0, ls='dotted', c='k')
    ax[1].set(xlim=(-15, 15), yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')

    if show_gaussian:
        from scipy.optimize import curve_fit
        def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        mean = 0
        sigma = 1

        ax[1].plot(xax * conf.PIXEL_SCALE,psfmodel[int(np.shape(psfmodel)[1]/2)], 'r')
        popt,pcov = curve_fit(gaus,xax * conf.PIXEL_SCALE,psfmodel[int(np.shape(psfmodel)[1]/2)],p0=[1,mean,sigma])

        ax[1].plot(xax*conf.PIXEL_SCALE,gaus(xax*conf.PIXEL_SCALE,*popt),'green')

    x = xax
    y = x.copy()
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + xv**2)
    cumcurve = [np.sum(psfmodel[radius<i]) for i in np.arange(0, np.shape(psfmodel)[0]/2)]
    ax[2].plot(np.arange(0, np.shape(psfmodel)[0]/2) * conf.PIXEL_SCALE, cumcurve)

    fig.suptitle(band)

    figname = os.path.join(conf.PLOT_DIR, f'{band}_psf.pdf')
    logger.info(f'Saving figure: {figname}')                
    fig.savefig(figname)
    plt.close(fig)
