# This is a very quick launch script

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
from core import interface
import config as conf

# if len(sys.argv) == 2:
#     interface.tractor(int(sys.argv[1]))

# elif len(sys.argv) == 3:
#     interface.tractor(int(sys.argv[1]), blob_id=int(sys.argv[2]))

# def tractor_brick(bricknum):
#     # make the thing
bricknum = int(sys.argv[1])
interface.make_models(bricknum)

# force it
for band in conf.BANDS:
    interface.force_models(brick_id=bricknum, band=band, insert=True)