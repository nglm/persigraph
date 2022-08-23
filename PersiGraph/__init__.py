# We typically want:
#
# import PersiGraph as pg
#
# g = pg.PersistentGraph()
# fig, axes = pg.vis.plot_overview(g)

# We do that to skip the persistent graph folder (which itself
# skip the persistentgraph.py file and access the class directly)
from .persistentgraph import PersistentGraph