# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Functions to handle command-line input

Known Issues
------------
None


"""


import os
import sys
import time
from functools import partial
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))

from tractor import NCircularGaussianPSF, PixelizedPSF, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev, GalaxyShape
from tractor.pointsource import PointSource

from astropy.io import fits
from astropy.table import Table, Column, vstack, join
from astropy.wcs import WCS
import numpy as np
import pathos.multiprocessing as mp 
import pathos.parallel as pp
from functools import partial
import matplotlib.pyplot as plt
# import psutil
import weakref

from .brick import Brick
from .mosaic import Mosaic
from .utils import plot_blob, SimpleGalaxy, plot_blobmap

import config as conf
plt.ioff()


def make_psf(multiband_only=False, sing_band=None):

    if not multiband_only:
        # Detection
        if conf.VERBOSE: print(f'Making PSF for {conf.DETECTION_NICKNAME}')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)
        if conf.VERBOSE: print(f'Mosaic loaded for {conf.DETECTION_NICKNAME}')
        detmosaic._make_psf()
        if conf.VERBOSE: print(f'PSF made successfully for {conf.DETECTION_NICKNAME}')

    # Bands
    if single_band is not None:
        sbands = [single_band,]
    else:
        sbands = conf.BANDS

    for i, band in enumerate(sbands):

        overwrite = True
        if insert:
            overwrite=False
        if i > 0:
            overwrite = False

        if conf.VERBOSE: print(f'Making PSF for {conf.MULTIBAND_NICKNAME} band {band}')
        bandmosaic = Mosaic(band)
        bandmosaic._make_psf()

    return


def make_bricks(multiband_only=False, single_band=None, insert=False, skip_psf=False):

    if not multiband_only:
        # Detection
        if conf.VERBOSE: print('Making mosaic for detection')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)
        if not skip_psf: 
            detmosaic._make_psf()

        if conf.NTHREADS > 0:
            pass
            # BUGGY DUE TO MEM ALLOC
            # if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(detmosaic._make_brick, detection=True, overwrite=True), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print('Making bricks for detection (in serial)')
            for brick_id in np.arange(1, detmosaic.n_bricks()+1):
                detmosaic._make_brick(brick_id, detection=True, overwrite=True)


    # Bands
    if single_band is not None:
        sbands = [single_band,]
    else:
        sbands = conf.BANDS

    for i, band in enumerate(sbands):

        overwrite = True
        if insert:
            overwrite=False
        if i > 0:
            overwrite = False

        if conf.VERBOSE: print(f'Making mosaic for band {band}')
        bandmosaic = Mosaic(band)
        if not skip_psf: 
            bandmosaic._make_psf()

        if conf.NTHREADS > 0:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in parallel)')
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            pool.map(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in serial)')
            for brick_id in np.arange(1, bandmosaic.n_bricks()+1):
                bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)

    return

"""
def tractor(brick_id, detection=True, multiband=True, source_id=None, blob_id=None): # need to add overwrite args!

    # Send out list of bricks to each node to call this function!

    # Create detection brick
    tstart = time.time()
    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time.time()
    #try:
    detbrick.sextract(conf.DETECTION_NICKNAME, sub_background=True)
    if conf.VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')
    #except:
    #    if conf.VERBOSE: print(f'Detection brick #{brick_id} sextraction FAILED. ({time.time() - tstart:3.3f}s)')
    #    return

    # Cleanup
    tstart = time.time()
    detbrick.cleanup()
    if conf.VERBOSE: print(f'Detection brick #{brick_id} gained {detbrick.n_blobs} blobs with {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

    # Create and update multiband brick
    tstart = time.time()
    fbrick = stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, detection=False)
    fbrick.blobmap = detbrick.blobmap
    fbrick.segmap = detbrick.segmap
    fbrick.catalog = detbrick.catalog.copy()
    fbrick.add_columns()
    if conf.VERBOSE: print(f'Multiband brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        conf.PLOT = True
        if source_id is not None:
            blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        detblob, fblob = detbrick.make_blob(blob_id), fbrick.make_blob(blob_id)
        runblob(blob_id, detblob, fblob, plotting=conf.PLOT)

    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = detbrick.n_blobs
        
        itstart = time.time()
        detblobs = [detbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        fblobs = [fbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        if conf.VERBOSE: print(f'Blobs created in {time.time() - itstart:3.3f}s')

        if conf.NTHREADS > 0:
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            #rows = pool.map(partial(runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
            
            #rows = pool.map(runblob, zip(detblobs, fblobs))
            
            output_rows = pool.map(runblob, np.arange(1, run_n_blobs+1), detblobs, fblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id, detblobs[kblob_id-1], fblobs[kblob_id-1], plotting=conf.PLOT) for kblob_id in np.arange(1, run_n_blobs+1)]
    
        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        output_rows = [x for x in output_rows if x is not None]

        output_cat = vstack(output_rows)
                
        for colname in output_cat.colnames:
           if colname not in fbrick.catalog.colnames:
               fbrick.catalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
        #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
        for row in output_cat:
            fbrick.catalog[np.where(fbrick.catalog['source_id'] == row['source_id'])[0]] = row

        # write out cat
        fbrick.catalog['x'] = detbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
        fbrick.catalog['y'] = detbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
        fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits')

        return
"""
        

def runblob(blob_id, blobs, detection=None, catalog=True, plotting=False):

    if conf.VERBOSE: print()
    if conf.VERBOSE: print(f'Starting on Blob #{blob_id}')
    tstart = time.time()

    detblob = None
    fblob = None
    if detection is None:
        detblob, fblob = weakref.proxy(blobs[0]), weakref.proxy(blobs[1])
    elif detection:
        detblob = weakref.proxy(blobs)
    else:
        fblob = weakref.proxy(blobs)

    conf.PLOT = plotting


    # Make blob with detection image 

    if detblob is not None:

        # Run models
        astart = time.time()
        detblob.stage_images()
        if conf.VERBOSE2: print(f'Images staged. ({time.time() - astart:3.3f})s')

        astart = time.time()
        status = detblob.tractor_phot()

        if not status:
            return detblob.bcatalog.copy()

        if conf.VERBOSE2: print(f'Morphology determined. ({time.time() - astart:3.3f})s')


        # Run follow-up phot
        if conf.DO_APPHOT:
            try:
                [[detblob.aperture_phot(band, img_type) for band in detblob.bands] for img_type in ('image', 'model', 'residual')]
            except:
                if conf.VERBOSE2: print(f'Aperture photmetry FAILED. Likely a bad blob.')
        if conf.DO_SEXPHOT:
            try:
                [detblob.sextract_phot(band) for band in detblob.bands]
            except:
                if conf.VERBOSE2: print(f'Residual Sextractor photmetry FAILED. Likely a bad blob.)')    

        duration = time.time() - tstart
        if conf.VERBOSE: print(f'Solution for Blob #{detblob.blob_id} (N={detblob.n_sources}) arrived at in {duration:3.3f}s ({duration/detblob.n_sources:2.2f}s per src)')
    
        catout = detblob.bcatalog.copy()
        del detblob

    if fblob is not None:
        # make new blob with band information
        astart = time.time() 

        if detblob is not None:
            fblob.model_catalog = detblob.solution_catalog.copy()
            fblob.position_variance = detblob.position_variance.copy()
            fblob.parameter_variance = detblob.parameter_variance.copy()
            if conf.VERBOSE2: print(f'Solution parameters transferred. ({time.time() - astart:3.3f})s')

        else:
            if catalog is None:
                raise ValueError('Input catalog not supplied!')
            else:
                blobmask = np.ones(len(catalog))
                if blob_id is not None:
                    blobmask = catalog['blob_id'] == blob_id
                fblob.bcatalog = catalog[blobmask]
                fblob.n_sources = len(fblob.bcatalog)
                catalog = catalog[blobmask]

                # print(catalog)
                # print(fblob.images[0].shape)
                # print(fblob.subvector)
                # print(fblob.mosaic_origin)
                # print(catalog['X_MODEL'], catalog['Y_MODEL'])
                catalog['X_MODEL'] -= fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                catalog['Y_MODEL'] -= fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1
                # print(catalog['X_MODEL'])
                # print(catalog['Y_MODEL'])

                fblob.model_catalog = models_from_catalog(catalog, fblob.bands, fblob.mosaic_origin)
                if (catalog['X_MODEL'] > fblob.images[0].shape[0]).any():
                    print('FAILED - BAD MODEL POSITION')
                    return fblob.bcatalog.copy()
                    # raise ValueError('BAD MODEL POSITION - FIX')

                fblob.position_variance = None
                fblob.parameter_variance = None
                fblob.bcatalog = catalog
                fblob.n_sources = len(catalog)

        # Forced phot
        astart = time.time() 
        fblob.stage_images()
        if conf.VERBOSE2: print(f'{fblob.bands} images staged. ({time.time() - astart:3.3f})s')

        astart = time.time() 
        status = fblob.forced_phot()

        if not status:
            return fblob.bcatalog.copy()

        if conf.VERBOSE2: print(f'Force photometry complete. ({time.time() - astart:3.3f})s')


        # Run follow-up phot
        if conf.DO_APPHOT:
            try:
                [[fblob.aperture_phot(band, img_type) for band in fblob.bands] for img_type in ('image', 'model', 'residual')]
            except:
                if conf.VERBOSE2: print(f'Aperture photmetry FAILED. Likely a bad blob.')
        if conf.DO_SEXPHOT:
            try:
                [fblob.sextract_phot(band) for band in fblob.bands]
            except:
                if conf.VERBOSE2: print(f'Residual Sextractor photmetry FAILED. Likely a bad blob.)')

        duration = time.time() - tstart
        if conf.VERBOSE: print(f'Solution for blob {fblob.blob_id} (N={fblob.n_sources}) arrived at in {duration:3.3f}s ({duration/fblob.n_sources:2.2f}s per src)')


        catout = fblob.bcatalog.copy()
        del fblob

    return catout


def make_models(brick_id, source_id=None, blob_id=None, segmap=None, catalog=None):
    # Create detection brick
    tstart = time.time()
    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time.time()
    #try:
    if (segmap is None) & (catalog is None):
        detbrick.sextract(conf.DETECTION_NICKNAME, sub_background=True, use_mask=False)
        if conf.VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')
    #except:
    #    if conf.VERBOSE: print(f'Detection brick #{brick_id} sextraction FAILED. ({time.time() - tstart:3.3f}s)')
    #    return
    elif (segmap is not None) & (catalog is not None):
        detbrick.catalog = catalog
        detbrick.segmap = segmap
    else:
        raise ValueError('No valid segmap and catalog provided to override SExtraction!')

    # Cleanup
    tstart = time.time()
    detbrick.cleanup()
    detbrick.add_columns()
    if conf.VERBOSE: print(f'Detection brick #{brick_id} gained {detbrick.n_blobs} blobs with {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

    if conf.PLOT:
        plot_blobmap(detbrick)

    # Save segmap and blobmaps
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=detbrick.segmap, name='SEGMAP'))
    hdul.append(fits.ImageHDU(data=detbrick.blobmap, name='BLOBMAP'))
    hdul.writeto(os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits'), overwrite=conf.OVERWRITE)
    hdul.close()

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        conf.PLOT = True
        outcatalog = detbrick.catalog.copy()
        mosaic_origin = detbrick.mosaic_origin
        brick_id = detbrick.brick_id
        if source_id is not None:
            blob_id = np.unique(detbrick.blobmap[detbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        detblob = detbrick.make_blob(blob_id)
        output_rows = runblob(blob_id, detblob, detection=True, plotting=conf.PLOT)

        output_cat = vstack(output_rows)
                
        for colname in output_cat.colnames:
            if colname not in outcatalog.colnames:
                outcatalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
        #outcatalog = join(outcatalog, output_cat, join_type='left', )
        for row in output_cat:
            outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

        # write out cat
        outcatalog['x'] = outcatalog['x'] + mosaic_origin[1] - conf.BRICK_BUFFER + 1.
        outcatalog['y'] = outcatalog['y'] + mosaic_origin[0] - conf.BRICK_BUFFER + 1.
        outcatalog.write(os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat'), format='fits', overwrite=conf.OVERWRITE)

    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = detbrick.n_blobs
        
        outcatalog = detbrick.catalog.copy()
        mosaic_origin = detbrick.mosaic_origin
        brick_id = detbrick.brick_id

        #detblobs = [detbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        detblobs = (detbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))
        #del detbrick

        if conf.NTHREADS > 0:

            #pool = pp.ParallelPool(processes=conf.NTHREADS)
            with mp.ProcessPool(processes=conf.NTHREADS) as pool:
                result = pool.map(partial(runblob, detection=True), np.arange(1, run_n_blobs+1), detblobs)

                output_rows = [res for res in result]

            #rows = pool.map(runblob, zip(detblobs, fblobs))
            # output_rows = pool.map(partial(runblob, detection=True), np.arange(1, run_n_blobs+1), detblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id+1, kblob, detection=True, plotting=conf.PLOT) for kblob_id, kblob in enumerate(detblobs)]

        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        #output_rows = [x for x in output_rows if x is not None]

        output_cat = vstack(output_rows)
                
        for colname in output_cat.colnames:
            if colname not in outcatalog.colnames:
                outcatalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
        #outcatalog = join(outcatalog, output_cat, join_type='left', )
        for row in output_cat:
            outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

        # write out cat
        outcatalog['x'] = outcatalog['x'] + mosaic_origin[1] - conf.BRICK_BUFFER + 1.
        outcatalog['y'] = outcatalog['y'] + mosaic_origin[0] - conf.BRICK_BUFFER + 1.
        outcatalog.write(os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat'), format='fits', overwrite=conf.OVERWRITE)

        return
    

def force_models(brick_id, band=None, source_id=None, blob_id=None, insert=True):
    # Create and update multiband brick
    tstart = time.time()

    if conf.NTHREADS > 0:
        conf.NTHREADS = 0
        if conf.VERBOSE: print('WARNING - Multithreading not supported while forcing models!')

    # for fband in band:
    fbrick = stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, band=band, detection=False)

    if band is None:
        band = conf.BANDS.copy()
    fband = [band,]
    if conf.VERBOSE: print(f'{fband} brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    if conf.VERBOSE: print(f'Forcing models on {fband}')

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        conf.PLOT = True
        if source_id is not None:
            blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        fblob = fbrick.make_blob(blob_id)
        output_rows = runblob(blob_id, fblob, detection=False, catalog=fbrick.catalog, plotting=conf.PLOT)

        output_cat = vstack(output_rows)

    
        if insert & conf.OVERWRITE & (conf.NBLOBS==0):
            # open old cat
            path_mastercat = os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat')
            if os.path.exists(path_mastercat):
                mastercat = Table.read(path_mastercat)

                # find new columns
                newcols = np.in1d(output_cat.colnames, mastercat.colnames, invert=True)
                # make fillers
                for colname in np.array(output_cat.colnames)[newcols]:
                    mastercat.add_column(output_cat[colname])
                    # mastercat.add_column(Column(np.zeros(len(mastercat), dtype=output_cat[colname].dtype), name=colname))
                # for row in output_cat:
                #     mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]] = row
                # coordinate correction
                # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                # save
                mastercat.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits', overwrite=conf.OVERWRITE)
                if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')
        else:
                
            for colname in output_cat.colnames:
                if colname not in fbrick.catalog.colnames:
                    fbrick.catalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
            #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
            for row in output_cat:
                fbrick.catalog[np.where(fbrick.catalog['source_id'] == row['source_id'])[0]] = row

            mode_ext = conf.MULTIBAND_NICKNAME
            if fband is not None:
                if len(fband) == 1:
                    mode_ext = fband[0].replace(' ', '_')
                else:
                    mode_ext = conf.MULTIBAND_NICKNAME

            # write out cat
            fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
            fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
            fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), format='fits', overwrite=conf.OVERWRITE)
            if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to new {fbrick.bands} catalog file.')


    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = fbrick.n_blobs

        fblobs = (fbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))

        if conf.NTHREADS > 0:

            with mp.ProcessPool(processes=conf.NTHREADS) as pool:
                result = pool.map(partial(runblob, detection=False), np.arange(1, run_n_blobs+1), fblobs)

                output_rows = [res for res in result]

            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # #rows = pool.map(partial(runblob, fbrick=fbrick, fbrick=fbrick), np.arange(1, fbrick.n_blobs))
            
            # #rows = pool.map(runblob, zip(fblobs, fblobs))
            # output_rows = pool.map(partial(runblob, detection=False), np.arange(1, run_n_blobs+1), fblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id, fbrick.make_blob(kblob_id), detection=False, catalog=fbrick.catalog, plotting=conf.PLOT) for kblob_id in np.arange(1, run_n_blobs+1)]

        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        #output_rows = [x for x in output_rows if x is not None]

        output_cat = vstack(output_rows)

    
        if insert & conf.OVERWRITE & (conf.NBLOBS==0):
            # open old cat
            path_mastercat = os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat')
            if os.path.exists(path_mastercat):
                mastercat = Table.read(path_mastercat)

                # find new columns
                newcols = np.in1d(output_cat.colnames, mastercat.colnames, invert=True)
                # make fillers
                for colname in np.array(output_cat.colnames)[newcols]:
                    mastercat.add_column(output_cat[colname])
                    # mastercat.add_column(Column(np.zeros(len(mastercat), dtype=output_cat[colname].dtype), name=colname))
                # for row in output_cat:
                #     mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]] = row
                # coordinate correction
                # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                # save
                mastercat.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits', overwrite=conf.OVERWRITE)
                if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')
        else:
                
            for colname in output_cat.colnames:
                if colname not in fbrick.catalog.colnames:
                    fbrick.catalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
            #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
            for row in output_cat:
                fbrick.catalog[np.where(fbrick.catalog['source_id'] == row['source_id'])[0]] = row

            mode_ext = conf.MULTIBAND_NICKNAME
            if fband is not None:
                if len(fband) == 1:
                    mode_ext = fband[0].replace(' ', '_')
                else:
                    mode_ext = conf.MULTIBAND_NICKNAME

            # write out cat
            fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
            fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
            fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), format='fits', overwrite=conf.OVERWRITE)
            if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to new {fbrick.bands} catalog file.')

    return


def stage_brickfiles(brick_id, nickname='MISCBRICK', band=None, detection=False):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(conf.BRICK_DIR, f'B{brick_id}_N{nickname}_W{conf.BRICK_WIDTH}_H{conf.BRICK_HEIGHT}.fits')

    if detection:
        sbands = [nickname,]
    elif band is None:
        sbands = conf.BANDS
    else:
        sbands = [band,]

    if os.path.exists(path_brickfile):
        # Stage things
        images = np.zeros((len(sbands), conf.BRICK_WIDTH + 2 * conf.BRICK_BUFFER, conf.BRICK_HEIGHT + 2 * conf.BRICK_BUFFER))
        weights = np.zeros_like(images)
        masks = np.zeros_like(images, dtype=bool)

        # Loop over expected bands
        with fits.open(path_brickfile) as hdul_brick:

            # Attempt a WCS
            wcs = WCS(hdul_brick[1].header)

            # Stuff data into arrays
            for i, tband in enumerate(sbands):
                images[i] = hdul_brick[f"{tband}_{conf.IMAGE_EXT.upper()}"].data
                weights[i] = hdul_brick[f"{tband}_{conf.WEIGHT_EXT.upper()}"].data
                masks[i] = hdul_brick[f"{tband}_{conf.MASK_EXT.upper()}"].data
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands)))
    for i, band in enumerate(sbands):
        path_psffile = os.path.join(conf.PSF_DIR, f'snap_{band}.fits')
        if os.path.exists(path_psffile):
            with fits.open(path_psffile) as hdul:
                psfmodel = hdul[0].data
                if i == 0:
                    psfmodels = np.zeros((len(sbands), psfmodel.shape[0], psfmodel.shape[1]))
                psfmodels[i] = psfmodel
        else:
            raise ValueError(f'PSF model not found for {band}!')
            psfmodels[i] = -99

    if detection:
        images, weights, masks = images[0], weights[0], masks[0]

    newbrick = Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands), brick_id=brick_id)

    if not detection:
        catalog = Table.read(os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat'))
        newbrick.catalog = catalog
        newbrick.n_sources = len(catalog)
        newbrick.n_blobs = catalog['blob_id'].max()
        try:
            newbrick.add_columns(band_only=True)
        except:
            if conf.VERBOSE: print('WARNING - could not add new columns. Overwrting old ones!')
        hdul_seg = fits.open(os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits'))
        newbrick.segmap = hdul_seg['SEGMAP'].data
        newbrick.blobmap = hdul_seg['BLOBMAP'].data 

    return newbrick


def models_from_catalog(catalog, band, rmvector):
        # make multiband catalog from det output

        model_catalog = -99 * np.ones(len(catalog), dtype=object)
        for i, src in enumerate(catalog):

            position = PixPos(src['X_MODEL'], src['Y_MODEL'])
            flux = Fluxes(**dict(zip(band, src['FLUX_DETECTION'] * np.ones(len(band)))))

            # # QUICK FIX
            # gamma = src['AB'].copy()
            # src['AB'] = (gamma**2) / (2 - (gamma**2))

            #shape = GalaxyShape(src['REFF'], 1./src['AB'], src['theta'])
            shape = EllipseESoft.fromRAbPhi(src['REFF'], 1./src['AB'], -src['THETA'])

            if src['SOLMODEL'] == 'PointSource':
                model_catalog[i] = PointSource(position, flux)
                model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
            elif src['SOLMODEL'] == 'SimpleGalaxy':
                model_catalog[i] = SimpleGalaxy(position, flux)
            elif src['SOLMODEL'] == 'ExpGalaxy':
                model_catalog[i] = ExpGalaxy(position, flux, shape)
            elif src['SOLMODEL'] == 'DevGalaxy':
                model_catalog[i] = DevGalaxy(position, flux, shape)
            elif src['SOLMODEL'] == 'FixedCompositeGalaxy':
                expshape = EllipseESoft.fromRAbPhi(src['EXP_REFF'], 1./src['EXP_AB'],  -src['EXP_THETA'])
                devshape = EllipseESoft.fromRAbPhi(src['DEV_REFF'], 1./src['DEV_AB'],  -src['DEV_THETA'])
                model_catalog[i] = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(src['FRACDEV']),
                                                expshape, devshape)

        return model_catalog