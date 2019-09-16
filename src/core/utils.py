"""

Filename: utils.py

Purpose: Set of utility functions

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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

import config as conf
import matplotlib.cm as cm
import random

colors = cm.rainbow(np.linspace(0, 1, 1000))
cidx = np.arange(0, 1000)
random.shuffle(cidx)
colors = colors[cidx]

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((h, w), dtype=int)
    mask[dist_from_center <= radius] = 1
    return mask


class SimpleGalaxy(ExpGalaxy):
    '''This defines the 'SIMP' galaxy profile -- an exponential profile
    with a fixed shape of a 0.45 arcsec effective radius and spherical
    shape.  It is used to detect marginally-resolved galaxies.
    '''
    shape = EllipseE(0.45 / conf.PIXEL_SCALE, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) + ')')

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'SimpleGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(SimpleGalaxy, self).isParamFrozen(pname)   

def plot_background(brick, idx, band=''):
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(brick.background_images[idx], cmap='Greys', norm=LogNorm())
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_background.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    if conf.VERBOSE2: print(f'Saving figure: {out_path}')

def plot_brick(brick, idx, band=''):
    fig, ax = plt.subplots(figsize=(20,20))
    backlevel, noisesigma = brick.backgrounds[idx]
    vmin, vmax = np.max([backlevel + noisesigma, 1E-5]), brick.images[idx].max()
    if vmin > vmax:
        print(f'WARNING - {band} brick not plotted!')
        return
    norm = LogNorm(vmin, vmax, clip='True')
    ax.imshow(brick.images[idx], cmap='Greys', origin='lower', norm=norm)
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_{band}_brick.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    if conf.VERBOSE2: print(f'Saving figure: {out_path}')

def plot_blob(myblob, myfblob):

    fig, ax = plt.subplots(ncols=4, nrows=1+myfblob.n_bands, figsize=(5 + 5*myfblob.n_bands, 10), sharex=True, sharey=True)
        
    back = myblob.backgrounds[0]
    mean, rms = back[0], back[1]
    noise = np.random.normal(mean, rms, size=myfblob.dims)
    tr = myblob.solution_tractor
    
    norm = LogNorm(np.max([mean + rms, 1E-5]), myblob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)

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
            
            norm = LogNorm(np.max([mean + rms, 1E-5]), myblob.images.max(), clip='True')
            img_opt = dict(cmap='Greys', norm=norm)

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
        print(f'COULD NOT PLOT MULTIBAND FITTING!')
                
    
    [[ax[i,j].set(xlim=(0,myfblob.dims[1]), ylim=(0,myfblob.dims[0])) for i in np.arange(myfblob.n_bands+1)] for j in np.arange(4)]
        
    #fig.suptitle(f'Solution for {blob_id}')
    fig.subplots_adjust(wspace=0.01, hspace=0, right=0.8)
    fig.savefig(os.path.join(conf.PLOT_DIR, f'{myblob.brick_id}_{myblob.blob_id}.pdf'))

def plot_detblob(blob, fig=None, ax=None, level=0, sublevel=0, final_opt=False, init=False):

    back = blob.backgrounds[0]
    mean, rms = back[0], back[1]
    noise = np.random.normal(mean, rms, size=blob.dims)
    tr = blob.solution_tractor
    
    norm = LogNorm(np.max([mean + rms, 1E-5]), blob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)

    # Init
    if fig is None:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(24,48), ncols=6, nrows=12)

        # Detection image
        ax[0,0].imshow(blob.images[0], **img_opt)
        [ax[0,i].axis('off') for i in np.arange(1, 6)]

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

    elif final_opt:
        nrow = 4 * level + 2* sublevel + 2
        [[ax[i,j].axis('off') for i in np.arange(nrow+1, 11)] for j in np.arange(0, 6)]

        ax[11,0].axis('off')
        residual = blob.images[0] - blob.solution_model_images[0]
        ax[11,1].imshow(blob.solution_model_images[0], **img_opt)
        ax[11,2].imshow(blob.solution_model_images[0] + noise, **img_opt)
        ax[11,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[11,4].imshow(blob.tr.getChiImage(0), cmap='RdGy', vmin = -5, vmax = 5)

        ax[11,1].set_ylabel('Solution')

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

        # Solution params
        for i, src in enumerate(blob.solution_catalog):
            ax[1,0].text(0.1, 0.8 - 0.4*i, f'#{blob.bcatalog[i]["source_id"]} Model:{src.name}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.1, f'       Flux: {src.brightness[0]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.2, f'       Chi2/N: {blob.solution_chisq[i,0]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)


        #fig.subplots_adjust(wspace=0, hspace=0)
        outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{conf.MODELING_NICKNAME}.pdf')
        fig.savefig(outpath)
        plt.close()
        if conf.VERBOSE2: print(f'Saving figure: {outpath}')

    else:
        if init:
            nrow = 4 * level + 2 * sublevel + 1
        else:
            nrow = 4 * level + 2 * sublevel + 2
        residual = blob.images[0] - blob.tr.getModelImage(0)
        ax[nrow,0].axis('off')
        ax[nrow,1].imshow(blob.tr.getModelImage(0), **img_opt)
        ax[nrow,2].imshow(blob.tr.getModelImage(0) + noise, **img_opt)
        ax[nrow,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[nrow,4].imshow(blob.tr.getChiImage(0), cmap='RdGy', vmin = -5, vmax = 5)

        models = {1:'PointSource', 3:'SimpleGalaxy', 5:'ExpGalaxy', 7:'DevGalaxy', 9:'CompositeGalaxy'}
        if init:
            ax[nrow,1].set_ylabel(models[nrow])
        
        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            
            if np.shape(residual) != np.shape(blob.segmap):
                print(np.shape(residual))
                print(np.shape(blob.segmap))
                plt.figure()
                plt.imshow(blob.segmap, cmap='Greys', norm=LogNorm())
                plt.savefig(os.path.join(conf.PLOT_DIR,'debug_segmap.pdf'))
                plt.figure()
                plt.imshow(residual, cmap='Greys', norm=LogNorm())
                plt.savefig(os.path.join(conf.PLOT_DIR,'debug_residual.pdf'))
                print('made debug plots!')
                print()
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[nrow,5].hist(res_seg, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
            if not init:
                ax[nrow,4].text(0.02, 0.9 - 0.1*i, r'$\chi^{2}$'+f'={blob.chisq[i, level, sublevel]:2.2f} | BIC={blob.bic[i, level, sublevel]:2.2f}',
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

    return fig, ax

def plot_fblob(blob, band, fig=None, ax=None, final_opt=False):

    idx = np.argwhere(blob.bands == band)[0][0]
    back = blob.backgrounds[idx]
    mean, rms = back[0], back[1]
    noise = np.random.normal(mean, rms, size=blob.dims)
    tr = blob.solution_tractor
    
    norm = LogNorm(np.max([mean + rms, 1E-5]), 0.90*blob.images.max(), clip='True')
    img_opt = dict(cmap='Greys', norm=norm)

    if final_opt:
        
        ax[2,0].axis('off')
        residual = blob.images[idx] - blob.solution_model_images[idx]
        ax[2,1].imshow(blob.solution_model_images[idx], **img_opt)
        ax[2,2].imshow(blob.solution_model_images[idx] + noise, **img_opt)
        ax[2,3].imshow(residual, cmap='RdGy', vmin=-5*rms, vmax=5*rms)
        ax[2,4].imshow(blob.tr.getChiImage(idx), cmap='RdGy', vmin = -5, vmax = 5)

        ax[2,1].set_ylabel('Solution')

        bins = np.linspace(np.nanmin(residual), np.nanmax(residual), 30)
        minx, maxx = 0, 0
        for i, src in enumerate(blob.bcatalog):
            res_seg = residual[blob.segmap==src['source_id']].flatten()
            ax[2,5].hist(res_seg, bins=20, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
        ax[2,5].set_xlim(minx, maxx)
        ax[2,5].axvline(0, c='grey', ls='dotted')
        ax[2,5].set_ylim(bottom=0)

        # Show chisq or rchisq?
        dof = ''
        if conf.USE_REDUCEDCHISQ:
            dof = '/N'

        # Solution params
        for i, src in enumerate(blob.solution_catalog):
            ax[1,0].text(0.1, 0.8 - 0.4*i, f'#{blob.bcatalog[i]["source_id"]} Model:{src.name}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.1, f'       Flux: {src.brightness[idx]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)
            ax[1,0].text(0.1, 0.8 - 0.4*i - 0.2, f'       Chi2{dof}: {blob.solution_chisq[i,0]:3.3f}', color=colors[i], transform=ax[1,0].transAxes)

        #fig.subplots_adjust(wspace=0, hspace=0)
        outpath = os.path.join(conf.PLOT_DIR, f'T{blob.brick_id}_B{blob.blob_id}_{blob.bands[idx]}.pdf')
        fig.savefig(outpath)
        plt.close()
        if conf.VERBOSE2: 
            print()
            print(f'Saving figure: {outpath}')

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
                e = Ellipse(xy=(objects['x'], objects['y']),
                            width=6*objects['a'],
                            height=6*objects['b'],
                            angle=objects['theta'] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor(colors[j])
                ax[0, 0].add_artist(e)

        ax[0,1].text(0.1, 0.9, f'Blob #{blob.blob_id}', transform=ax[0,1].transAxes)
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
            ax[1,5].hist(res_seg, bins=20, histtype='step', color=colors[i], density=True)
            resmin, resmax = np.nanmin(res_seg), np.nanmax(res_seg)
            if resmin < minx:
                minx = resmin
            if resmax > maxx:
                maxx = resmax
        ax[1,5].set_xlim(minx, maxx)
        ax[1,5].axvline(0, c='grey', ls='dotted')
        ax[1,5].set_ylim(bottom=0)

        [ax[1,i+1].set_title(title, fontsize=20) for i, title in enumerate(('Model', 'Model+Noise', 'Image-Model', '$\chi^{2}$', 'Residuals'))]

    
    return fig, ax

def plot_blobmap(brick):
    fig, ax = plt.subplots(figsize=(20,20))
    # imgs_marked = mark_boundaries(brick.images[0], brick.blobmap, color='red')[:,:,0]
    imgs_marked = find_boundaries(brick.blobmap, mode='thick').astype(int)
    imgs_marked[imgs_marked==0] = -99
    backlevel, noisesigma = brick.backgrounds[0]
    vmin, vmax = backlevel, backlevel + 5 * noisesigma
    norm = LogNorm(np.max([backlevel + noisesigma, 1E-5]), imgs_marked.max(), clip='True')
    ax.imshow(brick.images[0], cmap='Greys', origin='lower', norm=norm)
    mycmap = plt.cm.magma
    mycmap.set_under('k', alpha=0)
    ax.imshow(imgs_marked, alpha=0.9, cmap=mycmap, vmin=0, zorder=2, origin='lower')
    ax.scatter(brick.catalog['x'], brick.catalog['y'], marker='+', color='limegreen', s=0.1)
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
    out_path = os.path.join(conf.PLOT_DIR, f'B{brick.brick_id}_blobmaster.pdf')
    ax.axis('off')
    ax.margins(0,0)
    fig.savefig(out_path, dpi = 300, overwrite=True, pad_inches=0.0)
    plt.close()
    if conf.VERBOSE2: print(f'Saving figure: {out_path}')

def plot_ldac(tab_ldac, band, xlims=None, ylims=None, box=False):
    fig, ax = plt.subplots()
    ax.scatter(tab_ldac['FLUX_RADIUS'], tab_ldac['MAG_AUTO'], c='k', s=0.5)
    if box:
        rect = Rectangle((xlims[0], ylims[0]), xlims[1] - xlims[0], ylims[1] - ylims[0], fill=True, alpha=0.3,
                                edgecolor='r', facecolor='r', zorder=3, linewidth=1)
        ax.add_patch(rect)
    fig.subplots_adjust(bottom = 0.15)
    ax.set(xlabel='Flux Radius (px)', xlim=(0, 15),
            ylabel='Mag Auto (AB)', ylim=(26, 12))
    fig.savefig(os.path.join(conf.PLOT_DIR, f'{band}_box_{box}_ldac.pdf'), overwrite=True)

def plot_psf(psfmodel, band, show_gaussian=False):

    fig, ax = plt.subplots(ncols=2, figsize=(20,10))
    norm = LogNorm(1e-5, 0.1*np.nanmax(psfmodel), clip='True')
    img_opt = dict(cmap='Blues', norm=norm)
    ax[0].imshow(psfmodel, **img_opt, extent=0.15 *np.array([-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2, -np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2,]))
    ax[0].set(xlim=(-15,15), ylim=(-15, 15))
    ax[0].axvline(0, color='w', ls='dotted')
    ax[0].axhline(0, color='w', ls='dotted')

    xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2+0.5)
    [ax[1].plot(xax * 0.15, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[1])]
    ax[1].axvline(0, ls='dotted', c='k')
    ax[1].set(xlim=(-15, 15), yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')

    if show_gaussian:
        from scipy.optimize import curve_fit
        def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        mean = 0
        sigma = 1

        ax[1].plot(xax * 0.15,psfmodel[int(np.shape(psfmodel)[1]/2)], 'r')
        popt,pcov = curve_fit(gaus,xax * 0.15,psfmodel[int(np.shape(psfmodel)[1]/2)],p0=[1,mean,sigma])

        ax[1].plot(xax*0.15,gaus(xax*0.15,*popt),'green')

    figname = os.path.join(conf.PLOT_DIR, f'{band}_psf.pdf')
    if conf.VERBOSE2: print(f'Saving figure: {figname}')                
    fig.savefig(figname)
    plt.close(fig)
