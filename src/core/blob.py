# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class function to handle potentially blended sources (i.e. blobs)

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
from tractor import *
from time import time

from .subimage import Subimage
from .utils import SimpleGalaxy
from .config import *
# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
class Blob(Subimage):

    def __init__(self, brick, blob_id):
            # super().__init__(self.images, self.weights, self.masks, self.bands, self.wcs, self.subvector)

                
        # Grab blob
        self.blob_id = blob_id
        blobmask = np.array(brick.blobmap == self.blob_id, bool)
        blob_sources = np.unique(brick.segmap[blobmask])

        # Dimensions
        idx, idy = blobmask.nonzero()
        xlo, xhi = np.min(idx), np.max(idx) + 1
        ylo, yhi = np.min(idy), np.max(idy) + 1
        w = xhi - xlo
        h = yhi - ylo

        # Make cutout
        blob_comps = brick._get_subimage(xlo, ylo, w, h, buffer=BLOB_BUFFER)
        self.images, self.weights, self.masks, self.psfmodels, self.bands, self.wcs, self.subvector, self.slicepix, self.slice = blob_comps

        self.masks[self.slicepix] = np.logical_not(blobmask[self.slice], dtype=bool)
        self.segmap = brick.segmap[self.slice]

        # Clean
        blob_sourcemask = np.in1d(brick.catalog['sid'], blob_sources)
        self.catalog = brick.catalog[blob_sourcemask]
        self.catalog['x'] -= self.subvector[1]
        self.catalog['y'] -= self.subvector[0]
        self.n_sources = len(self.catalog)
            
        self.mids = np.ones(self.n_sources, dtype=int)
        self.model_catalog = np.zeros(self.n_sources, dtype=object)
        self.solution_catalog = np.zeros(self.n_sources, dtype=object)
        self.solved_chisq = np.zeros(self.n_sources)
        self.tr_catalogs = np.zeros((self.n_sources, 3, 2), dtype=object)
        self.chisq = np.zeros((self.n_sources, 3, 2))

        
    def stage_images(self):

        timages = np.zeros(self.n_bands, dtype=object)
        for i, (image, weight, mask, psf, band) in enumerate(zip(self.images, self.weights, self.masks, self.psfmodels, self.bands)):
            tweight = weight.copy()
            tweight[mask] = 0
            if psf is -99:
                psfmodel = NCircularGaussianPSF([2,], [1,])
            else:
                psfmodel = PixelizedPSF(psf)
            timages[i] = Image(data=image,
                            invvar=tweight,
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=LinearPhotoCal(1, band),
                            sky=ConstantSky(0.),
                            )
        self.timages = timages

    def stage_models(self):
        # Currently this makes NEW models for each trial. Do we want to freeze solution models and re-use them?

        # Trackers


        for i, (mid, src) in enumerate(zip(self.mids, self.catalog)):

            freeze_position = (self.mids == 2).all()
            if freeze_position:
                position = self.tr_catalogs[i,0,0].getPosition()
            else:
                position = PixPos(src['x'], src['y'])
            flux = Fluxes(**dict(zip(self.bands, src['flux'] * np.ones(self.n_bands))))
            shape = GalaxyShape(1, src['b'] / src['a'], src['theta'])

            if mid == 1:
                self.model_catalog[i] = PointSource(position, flux)
            elif mid == 2:
                self.model_catalog[i] = SimpleGalaxy(position, flux)
            elif mid == 3:
                self.model_catalog[i] = ExpGalaxy(position, flux, shape)
            elif mid == 4:
                self.model_catalog[i] = DevGalaxy(position, flux, shape)
            elif mid == 5:
                self.model_catalog[i] = galaxy.FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(0.5),
                                                shape, shape
                                                )
            if freeze_position:
                self.model_catalog[i].freezeParams('pos')               


    def tractor_phot(self):
        # Currently not forcing anything.
        idx_models = ((1,2), (3,4), (5,))

        self._solved = self.solution_catalog != 0
    
        self._level = -1
        while not self._solved.all():
            self._level += 1
            for sublevel in np.arange(len(idx_models[self._level])):
                self._sublevel = sublevel

                # prepare models
                self.mids[~self._solved] = idx_models[self._level][sublevel]
                self.stage_models()

                # store
                self.tr = Tractor(self.timages, self.model_catalog)

                # optimize
                self.status = self.optimize_tractor()

                if self.status == False:
                    return False

                # clean up
                self.tr_catalogs[:, self._level, self._sublevel] = self.tr.getCatalog()    

                for i, src in enumerate(self.catalog):
                    if self._solved[i]:
                        continue
                    totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == src['sid']])**2)
                    ### PENALTIES ARE TBD
                    # residual = self.images[0] - self.tr.getModelImage(0)
                    # if np.mean(residual[self.masks[0]]) < 0:
                    #     totalchisq = 1000000
                    self.chisq[i, self._level, self._sublevel] = totalchisq

                # Move unsolved to next sublevel
                if sublevel == 0:  
                    self.mids[~self._solved] += 1

            # decide
            self.decide_winners()
            self._solved = self.solution_catalog != 0

            print()
            print(self.chisq)
            print(self.solution_catalog)

        # Final optimization
        print()
        print('Final opt')
        self.model_catalog = self.solution_catalog
        self.tr = Tractor(self.timages, self.model_catalog)

        self.status = self.optimize_tractor()

        self.solution_tractor = self.tr
        self.solution_catalog = self.tr.getCatalog()

        for i, src in enumerate(self.catalog):
            totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == src['sid']])**2)
            self.solved_chisq[i] = totalchisq

        print()
        print(self.solved_chisq)
        print(self.solution_catalog[:])
            
        return True

    def decide_winners(self):
        # take the model_catalog and chisq and figure out what's what
        # Only look at unsolved models!

        # holders - or else it's pure insanity.
        chisq = self.chisq[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_chisq = self.solved_chisq[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG)?
            chmask = (chisq[:, 0, 0] < chisq[:, 0, 1]) 
            if chmask.any():
                solution_catalog[chmask] = tr_catalogs[chmask, 0, 0].copy()
                solved_chisq[chmask] = chisq[chmask, 0, 0]

            # So chi2(SG) is min, try more models
            mids[~chmask] = 2


        if self._level == 1:
            # For which are they nearly equally good?
            movemask = (abs(chisq[:, 1, 0] - chisq[:, 1, 1]) < EXP_DEV_THRESH) 

            # Has Exp beaten SG?
            expmask = (chisq[:, 1, 0] < chisq[:, 0, 1]) 

            # Has Dev beaten SG?
            devmask = (chisq[:, 1, 1] < chisq[:, 0, 1]) 

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask
            
            # For which was SG better
            premask = ~expmask & ~devmask

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (chisq[:, 1, 0] < chisq[:, 1, 1])

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (chisq[:, 1, 1] < chisq[:, 1, 0])

            if nextmask.any():
                mids[nextmask] = 4
                
            if premask.any():
                solution_catalog[premask] = tr_catalogs[premask, 0, 1].copy()
                solved_chisq[premask] = chisq[premask, 0, 1]  

            if nexpmask.any():
                
                solution_catalog[nexpmask] = tr_catalogs[nexpmask, 1, 0].copy()
                solved_chisq[nexpmask] = chisq[nexpmask, 1, 0] 
           
            if ndevmask.any():
                
                solution_catalog[ndevmask] = tr_catalogs[ndevmask, 1, 1].copy()
                solved_chisq[ndevmask] = chisq[ndevmask, 1, 1]


        if self._level == 2:
            # For which did Comp beat EXP and DEV?
            compmask = (chisq[:, 2, 0] < chisq[:, 1, 0]) &\
                       (chisq[:, 2, 0] < chisq[:, 1, 1]) 

            print(compmask)
    
            if compmask.any():
                solution_catalog[compmask] = tr_catalogs[compmask, 2, 0].copy()
                solved_chisq[compmask] = chisq[compmask, 2, 0]
        
            # where better as EXP or DEV
            if (~compmask).any():
                ch_exp = (chisq[:, 1, 0] < chisq[:, 1, 1]) & ~compmask

                if ch_exp.any():
                    solution_catalog[ch_exp] = tr_catalogs[ch_exp, 1, 0].copy()
                    solved_chisq[ch_exp] = chisq[ch_exp, 1, 0]

                ch_dev = (chisq[:, 1, 1] < chisq[:, 1, 0]) & ~compmask

                if ch_dev.any():
                    solution_catalog[ch_dev] = tr_catalogs[ch_dev, 1, 1].copy()
                    solved_chisq[ch_dev] = chisq[ch_dev, 1, 1]

        # hand back
        self.chisq[~self._solved] = chisq
        self.solution_catalog[~self._solved] = solution_catalog
        self.solved_chisq[~self._solved] = solved_chisq
        self.mids[~self._solved] = mids


    def optimize_tractor(self, tr=None):

        if tr is None:
            tr = self.tr 

        tr.freezeParams('images')
        #tr.thawAllParams()

        start = time()
        for i in range(TRACTOR_MAXSTEPS):
            try:
                
                dlnp, __, __ = tr.optimize()
                print('[blob.fit_morph] :: dlnp {}'.format(dlnp))
            except:
                print('FAILED')
                return False

            if dlnp < TRACTOR_CONTHRESH:
                break

        return True


    def aperture_phot(self):
        # photutils
        pass

    def sextract_phot(self):
        # Allow user to enter image (i.e. image, residual, model...)
        pass

    def forced_phot(self):
        
        # Update the incoming models
        for i, model in enumerate(self.model_catalog):
            model.brightness = Fluxes(**dict(zip(self.bands, model.brightness[0] * np.ones(self.n_bands))))   
            model.freezeAllBut('brightness')     

        # Stash in Tractor
        self.tr = Tractor(self.timages, self.model_catalog)

        # Optimize
        status = self.optimize_tractor()

        # Chisq
        self.solution_chisq = np.zeros((self.n_sources, self.n_bands))
        for i, src in enumerate(self.catalog):
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['sid']])**2)
                ### PENALTIES ARE TBD
                # residual = self.images[j] - self.tr.getModelImage(j)
                # if np.median(residual[self.masks[j]]) < 0:
                #     totalchisq = 1E30
                self.solution_chisq[i, j] = totalchisq

        self.solution_tractor = Tractor(self.timages, self.tr.getCatalog())
        self.solution_catalog = self.solution_tractor.getCatalog()

        return status