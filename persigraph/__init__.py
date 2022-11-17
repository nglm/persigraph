# We typically want:
#
# import persigraph as pg
#
# g = pg.PersistentGraph()
# fig, axes = pg.plots.overview(g)
# SCORES = pg.SCORES
# CLUSTERING_METHODS = pg.CLUSTERING_METHODS

# We do that to skip the persistentgraph folder (which itself
# skip the persistentgraph.py file and access the class directly)
from .persistentgraph import PersistentGraph
from .persistentgraph import plots, analysis

# Select which modules should be available to the user
from . import vis
from . import datasets
from . import utils