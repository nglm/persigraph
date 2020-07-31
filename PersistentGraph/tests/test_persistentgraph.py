import pytest


# ------
# Source
# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to access persistentgraph
# ------

from persitentgraph import PersistentGraph

