# We typically want:
#
# import PersiGraph as pg
#
# g = pg.PersistentGraph()
# fig, axes = pg.plot_overview(g)

# We do that to skip the persistent graph folder (which itself
# skip the persistentgraph.py file and access the class directly)
from .persistentgraph import PersistentGraph
from .persistentgraph import plots, analysis

# Select which modules should be available to the user
from . import vis
from . import datasets
