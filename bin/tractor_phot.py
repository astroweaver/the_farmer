# This is a very quick launch script

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
from core import interface

if len(sys.argv) == 1:
    interface.tractor(int(sys.argv[1]))

elif len(sys.argv) == 2:
    interface.tractor(int(sys.argv[1]), blob_id=(sys.argv[2]))