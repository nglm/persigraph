PersiGraph
===============================================================================

**PersiGraph is python a package specialized in clustering ensemble of time-series**, notably when the ensemble size is small such as in ensemble weather prediction.

PersiGraph generates a graph where the vertices represent the different clusters and the edges the evolution of the clusters with time. It allows for a **dynamic number of cluster, meaning that clusters can merge and split with time**.

In addition to a fully automated generation of the most relevant graph of clusters, PersiGraph aims at **taking into account the potential uncertainty around the number of clusters when building the graph**, so that the user can get a more faithful representation of the data and can make an informed decision on the final number of clusters at each time step. **To assess the quality of the clusters, PersiGraph relies on the package [PyCVI](https://github.com/nglm/pycvi)**.

Install
-------------------------------------------------------------------------------

In a poetry project, run:

```bash
poetry add git+https://github.com/nglm/persigraph.git
```
