# To replace 'from .component import Component'
# And use from . import Component'

from .component import Component
from .vertex import Vertex
from .edge import Edge
from .persistentgraph import PersistentGraph

# Select some constants
# We typically want:
#
# import persigraph as pg
# SCORES = pg.SCORES
# CLUSTERING_METHODS = pg.CLUSTERING_METHODS
from ._clustering_model import CLUSTERING_METHODS
from ._scores import SCORES

# Select which modules should be available to the user
# from . import plots
