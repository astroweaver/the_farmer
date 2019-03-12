# This is a very quick launch script

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
from core import interface


interface.tractor(sys.argv[1])