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
import matplotlib.pyplot as plt

import config as conf
import matplotlib.cm as cm
import random

colors = cm.rainbow(np.linspace(0, 1, 100))
cidx = np.arange(0, 100)
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
    shape = EllipseE(0.45, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)
        self.shape = SimpleGalaxy.shape

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


def plot_blob(myblob, myfblob):

    fig, ax = plt.subplots(ncols=4, nrows=1+myfblob.n_bands, figsize=(5 + 5*myfblob.n_bands, 10), sharex=True, sharey=True)
        
    back = myblob.backgrounds[0]
    mean, rms = back.globalback, back.globalrms
    noise = np.random.normal(mean, rms, size=myfblob.dims)
    tr = myblob.solution_tractor
    
    img_opt = dict(cmap='Greys', vmin = mean, vmax = mean + 10 * rms)

    ax[0, 0].imshow(myblob.images[0], **img_opt)
    ax[0, 1].imshow(tr.getModelImage(0) + noise, **img_opt)
    ax[0, 2].imshow(myblob.images[0] - tr.getModelImage(0), cmap='RdGy', vmin=-10*rms, vmax=10*rms)    
    ax[0, 3].imshow(tr.getChiImage(0), cmap='RdGy', vmin = -7, vmax = 7)
    
    ax[0, 0].set_ylabel(f'Detection ({myblob.bands[0]})')
    ax[0, 0].set_title('Data')
    ax[0, 1].set_title('Model + Noise')
    ax[0, 2].set_title('Data - Model')
    ax[0, 3].set_title('$\chi$-map')
    
    band = myblob.bands[0]
    for j, src in enumerate(myblob.solution_catalog):
        mtype = src.name
        flux = src.brightness[0]
        chisq = myblob.solved_chisq[j]
        topt = dict(color=colors[j], transform = ax[0, 3].transAxes)
        ystart = 0.99 - j * 0.4
        ax[0, 3].text(1.05, ystart - 0.1, f'{j}) {mtype}', **topt)
        ax[0, 3].text(1.05, ystart - 0.2, f'  F({band}) = {flux:4.4f}', **topt)
        ax[0, 3].text(1.05, ystart - 0.3, f'  $\chi^{2}$ = {chisq:4.4f}', **topt)

    for i in np.arange(myfblob.n_bands):
        back = myfblob.backgrounds[i]
        mean, rms = back.globalback, back.globalrms
        noise = np.random.normal(mean, rms, size=myfblob.dims)
        tr = myfblob.solution_tractor
        print(tr)
        
        img_opt = dict(cmap='Greys', vmin = mean, vmax = mean + 10 * rms)
        ax[i+1, 0].imshow(myfblob.images[i], **img_opt)
        ax[i+1, 1].imshow(tr.getModelImage(i) + noise, **img_opt)
        ax[i+1, 2].imshow(myfblob.images[i] - tr.getModelImage(i), cmap='RdGy', vmin=-10*rms, vmax=10*rms)  
        ax[i+1, 3].imshow(tr.getChiImage(i), cmap='RdGy', vmin = -7, vmax = 7)
        
        ax[i+1, 0].set_ylabel(myfblob.bands[i])
        
        band = myfblob.bands[i]
        for j, src in enumerate(myfblob.solution_catalog):
            mtype = src.name
            flux = src.brightness[i]
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
                
        
        
    #fig.suptitle(f'Solution for {blob_id}')
    fig.subplots_adjust(wspace=0.01, hspace=0, right=0.8)
    fig.savefig(os.path.join(conf.PLOT_DIR, f'{myblob.brick_id}_{myblob.blob_id}.png'))
    plt.pause(0.1)
