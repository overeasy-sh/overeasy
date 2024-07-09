from overeasy.agents import *
from overeasy.models import *

__version__ = "0.2.6"


import os as _os
ROOT = _os.path.expanduser("~/.overeasy")
_os.makedirs(ROOT, exist_ok=True)