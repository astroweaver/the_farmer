# take list of sources and a band
# loop over the bricks, modify segs/blobs, run only affected blobs in fphot
# grab new sources + affected blobs and make VAC, marking which ones are 'new'
# for now, don't update the models.

# set up so I can run it on CANDIDE...


# Configuration
import numpy as np 
import sys
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from matplotlib.colors import LogNorm

plt.ioff()
# sys.path.insert(0, os.path.join('/n07data/weaver/COSMOS2020/config'))
# import config as conf

# FAKE THIS FOR NOW!

class foo_config:
    def __init__(self):
        self.BRICK_WIDTH = 2004    																			# Width of bricks (in pixels)
        self.BRICK_HEIGHT = 2004   																			# Height of bricks (in pixels)
        self.BRICK_BUFFER = 100																				# Buffer around bricks (in pixels)
        self.BLOB_BUFFER = 5 #px																				# Buffer around blobs (in pixels)
        self.DILATION_RADIUS = 5
        self.PIXEL_SCALE = 0.15																				# Image pixel scale in arcsec/px^2
        self.MOSAIC_WIDTH = 48096																			# Mosaic image width
        self.MOSAIC_HEIGHT = 48096	
        self.dims = (2204, 2204)
        INPUT_DIR = '/n07data/weaver/COSMOS2020/'
        self.IN_BRICK_DIR = INPUT_DIR+'/data/intermediate/bricks'
        self.IN_INTERIM_DIR = INPUT_DIR+'/data/intermediate/interim/cosmos_chimeanmodel4'
        self.IN_CATALOG_DIR = INPUT_DIR+'/data/output/catalogs/cosmos_chimeanmodel4'					# OUTPUT Catalog

        WORKING_DIR = '/n07data/weaver/COSMOS2020/data_iracout'
        # self.IMAGE_DIR = WORKING_DIR+'/data/images'										# INPUT Raw images, weights, and masks
        # self.PSF_DIR = INPUT_DIR+'/data/intermediate/psfmodels'							# INTERMEDIATE Point-spread function files
        self.BRICK_DIR = WORKING_DIR+'/intermediate/bricks'
        self.INTERIM_DIR = WORKING_DIR+'/intermediate/interim'
        self.CATALOG_DIR = WORKING_DIR+'/output/catalogs'						# INTERMEDIATE Bricks (i.e. subimages)
        # BRICK_DIR = '/Volumes/LaCie12/Data/COSMOS/mosaics/brick_tests'
        						# OUTPUT Other stuff I make 		
        self.PLOT_DIR = WORKING_DIR+'/intermediate/plots'							# OUTPUT Figures
conf = foo_config()

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

def _get_origin(brick_id, brick_width=conf.BRICK_WIDTH, brick_height=conf.BRICK_HEIGHT):
        x0 = int(((brick_id - 1) * brick_width) % conf.MOSAIC_WIDTH)
        y0 = int(((brick_id - 1) * brick_height) / conf.MOSAIC_HEIGHT) * brick_height
        return np.array([x0, y0])
        

INPUT_CAT = '/n07data/weaver/COSMOS2020/data_iracout/irac-resid_v2_visual-class.fits'
MASTER_CAT = '/n07data/COSMOS2020/RELEASES/R1_v2.0_20210625/COSMOS2020_FARMER_R1_v2.0.fits'

SHOW_PLOT = False

# Read catalog
input_cat = Table.read(INPUT_CAT)
master_cat = Table.read(MASTER_CAT)

# find out which sources are in which bricks...
# I guess I just loop over and find neighbors... Maybe we get lucky.

incoord = SkyCoord(input_cat['ALPHA_J2000'], input_cat['DELTA_J2000'], unit=u.deg)
mastcoord = SkyCoord(master_cat['ALPHA_J2000'], master_cat['DELTA_J2000'], unit=u.deg)

print('Computing matches to get brick membership')
idx, d2d, d3d = incoord.match_to_catalog_sky(mastcoord)

objids = master_cat['ID'][idx]
bricks = [int(i.split('_')[0]) for i in master_cat['FARMER_ID'][idx]]

ubricks, nbricks = np.unique(bricks, return_counts=True)
master_bids = []
# Loop over bricks...
print(f'Found {len(ubricks)} unique bricks for {len(incoord)} sources.')
for brickid, num in zip(ubricks, nbricks):
    # if brickid != 79: continue
    print(brickid, num)

    # brick, for wcs
    hdul = fits.open(f'{conf.IN_BRICK_DIR}/B{brickid}_NMULTIBAND_W2004_H2004.fits')

    ch1 = hdul['IRAC_CH1_IMAGE']

    from astropy.wcs import WCS
    wcs = WCS(hdul[1].header)

    hdul = fits.open(f'{conf.IN_INTERIM_DIR}/B{brickid}_SEGMAPS.fits')
    segmap, blobmap = hdul['SEGMAP'].data, hdul['BLOBMAP'].data

    tab = Table.read(f'{conf.IN_CATALOG_DIR}/B{brickid}.cat')

    bids = []
    iids = []
    fids = []
    for obj, coord, neighid, sep in zip(input_cat[bricks==brickid], incoord[bricks==brickid], objids[bricks==brickid], d2d[bricks==brickid]):

        iid = obj['NUMBER']
        neighbor = master_cat[master_cat['ID']==neighid][0]
        id1 = neighbor['ID']
        bid, sid = neighbor['FARMER_ID'].split('_')
        mcoord = mastcoord[master_cat['ID']==neighid]

        iids.append(iid)


        print(f'Found match within {sep.to(u.arcsec):2.2f}')
        print(f'N={iid}: Matched object to #{id1} within brick {bid} (sid {sid})')

        # Probably need something to catch a duplication...
        if sep < 1.0*u.arcsec:
            print('Already detected!')
            continue

        max_sid = tab['source_id'].max()

        cx, cy = coord.to_pixel(wcs)

        tab.add_row()
        tab[-1]['source_id'] = max_sid + 1
            
        tab[-1]['brick_id'] = brickid
        tab[-1]['RA_DETECTION'] = coord.ra.to(u.deg).value
        tab[-1]['DEC_DETECTION'] = coord.dec.to(u.deg).value
        pixx, pixy = coord.to_pixel(wcs)
        origin = _get_origin(brickid)
        pixx += origin[1] + conf.BRICK_BUFFER
        pixy += origin[0] + conf.BRICK_BUFFER
        tab[-1]['x'] = pixx
        tab[-1]['y'] = pixy
        tab[-1]['x_orig'] = pixx
        tab[-1]['y_orig'] = pixy
        tab[-1]['a'] = 1
        tab[-1]['b'] = 1

        # tab['x'] -= origin[1] - conf.BRICK_BUFFER
        # tab['y'] -= origin[0] - conf.BRICK_BUFFER
        tab[-1]['x_orig'] -= 2*conf.BRICK_BUFFER
        tab[-1]['y_orig'] -= 2*conf.BRICK_BUFFER


        # Model info!
        tab[-1]['VALID_SOURCE_MODELING'] = True
        tab[-1]['SOLMODEL_MODELING'] = 'PointSource'
        tab[-1]['BEST_MODEL_BAND'] = 'MODELING'
        tab[-1]['X_MODEL'] = tab[-1]['x_orig']
        tab[-1]['Y_MODEL'] = tab[-1]['y_orig']
        tab[-1]['X_MODEL_MODELING'] = tab[-1]['x_orig']
        tab[-1]['Y_MODEL_MODELING'] = tab[-1]['y_orig']
        tab[-1]['Y_MODEL'] = tab[-1]['y_orig']
        tab[-1]['RA'] = tab[-1]['RA_DETECTION']
        tab[-1]['DEC'] = tab[-1]['DEC_DETECTION']
        
        print('Updated table.')

        # put down segs, blobs
        sh, sw = np.shape(segmap)
        cx, cy = coord.to_pixel(wcs)
        mask = create_circular_mask(sh, sw, center=(cx, cy), radius = 5)

        segmap[mask==1] = tab[-1]['source_id']
        fids.append(tab[-1]['source_id'])

        mask = create_circular_mask(sh, sw, center=(cx, cy), radius = 5+conf.DILATION_RADIUS)
        if (blobmap[mask==1] == 0).all():
            # make a new blob!
            print('Making NEW blob!')
            tab[-1]['N_BLOB'] = 1
            tab[-1]['blob_id'] = np.nanmax(tab['blob_id']) + 1
            blobmap[mask==1] = tab[-1]['blob_id']
        else:
            neighborblob, counts = np.unique(blobmap[mask==1], return_counts=True)
            neighborblob, counts = neighborblob[neighborblob>0], counts[neighborblob>0]
            choice = neighborblob[counts == np.max(counts)]
            blobmap[(blobmap==0) & (mask==1)] = choice
            if neighbor['VALID_SOURCE'] == False:
                print('ERROR -- neighbor does NOT have a valid source.')
                continue
            tab[-1]['N_BLOB'] = neighbor['N_GROUP'] + 1
            tab[tab['blob_id'] == choice]['N_BLOB'] += 1
            tab[-1]['blob_id'] = choice
            print(f'Adding to nearby blob {choice} (N_BLOB = {tab[-1]["N_BLOB"]})')


        bids.append(tab[-1]['blob_id'])
            
        from astropy.nddata.utils import Cutout2D

        subimg = Cutout2D(ch1.data, coord, 15*u.arcsec, wcs)
        subseg = Cutout2D(segmap, coord, 15*u.arcsec, wcs)
        subblob = Cutout2D(blobmap, coord, 15*u.arcsec, wcs)

        if SHOW_PLOT:
            print('plot.')
            fig, ax = plt.subplots(ncols=3, figsize=(10, 10))
            ax[0].imshow(subimg.data, cmap='Greys', norm=LogNorm(1e-5, 5e-3))

            from matplotlib import colors
            color = plt.get_cmap('rainbow', len(np.unique(subseg.data)))
            for i, segid in enumerate(np.unique(subseg.data)):
                fill = subseg.data.copy()
                fill = fill.astype('float')
                fill[fill!=segid] = np.nan
                cmap = colors.ListedColormap([color(i)])

                ax[1].imshow(fill, cmap=cmap, vmin=3576, vmax=3634)

            color = plt.get_cmap('rainbow', len(np.unique(subblob.data)))
            for i, blobid in enumerate(np.unique(subblob.data)):
                fill = subblob.data.copy()
                fill = fill.astype('float')
                fill[fill!=blobid] = np.nan
                cmap = colors.ListedColormap([color(i)])

                ax[2].imshow(fill, cmap=cmap, vmin=np.nanmin(fill), vmax=np.nanmax(fill))


            for row in tab:
                coord2 = SkyCoord(row['RA_DETECTION'], row['DEC_DETECTION'], unit=u.deg)
                separation = coord.separation(coord2)
                if separation < 8*u.arcsec:
                    px, py = coord2.to_pixel(subimg.wcs)
                    ax[0].scatter(px, py, c='w')
                    ax[1].scatter(px, py, c='w')
                    ax[2].scatter(px, py, c='w')
            fig.savefig(conf.PLOT_DIR+f'/{iid}_B{bid}.pdf')

    # remove any untouched sources
    keep = np.zeros(len(tab), dtype=bool)
    for iid, fid, bid in zip(iids, fids, bids):
        keep[tab['blob_id'] == bid] = True
        master_bids.append((iid, fid, bid, brickid)) #input ID, farmer new ID, blob ID, brick ID

    print(f'Keeping {np.sum(keep)} sources in blobs {bids}')
    tab = tab[keep]

    # Update files
    tab.write(f'{conf.CATALOG_DIR}/B{brickid}.cat', format='fits', overwrite=True)
    hdul = fits.open(f'{conf.IN_INTERIM_DIR}/B{brickid}_SEGMAPS.fits')
    hdul['SEGMAP'].data = segmap
    hdul['BLOBMAP'].data = blobmap
    hdul.writeto(f'{conf.INTERIM_DIR}/B{brickid}_SEGMAPS.fits', overwrite=True)
    
# Save the input IDs and brick IDs
with open(f'{conf.INTERIM_DIR}/masterbids.npy', 'wb') as f:
    np.save(f, master_bids)

# # Now that everything is prepped, run Farmer on blobs
# from src.core import interface

# for iid, fid, bid, brickid in master_bids:
#     interface.force_photometry(brickid, blob_id=bid)

# # Lastly need to grab catalogs, isolate the ones we ran again, and combine them into a new catalog
# for iid, fid, bid, brickid in master_bids:
#     ctab = Table.read(DIR_OUTPUT+f'{brickid}_MULTIBAND.cat')
#     # ... stack