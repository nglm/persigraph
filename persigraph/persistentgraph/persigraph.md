PersiGraph
===============================================================================

Data
-------------------------------------------------------------------------------

The `PersiGraph` method is designed to cluster ensembles of multivariate time series $\mathcal{X} \in \mathbb{R}^{N \times d \times T}$, where $N$ is the number of members in the ensemble, $d$ the number of variables and $T$ the length of the time series. In order to emulate the case where there is no cluster ($k=0$), the concept of "*ensemble zero*" $\mathcal{X}_{zero}$ is introduced. *ensemble zero* is an ensemble that represents the behaviour of a uniform distribution with the same number of members and the same value ranges as the original data.

In the earliest versions of the `PersiGraph` method, only univariate time series could be considered (i.e., $d=1$). `PersiGraph` has now been extended to multivariate time series, i.e., with $d>=1$. In addition, so far, the data being clustered, the data used compute scores and the data used to create the graph components were the same. For each time step $t$, the subset $\mathbf{X_t}$ of $\mathcal{X}$ was extracted, $\mathbf{X}_t = \mathcal{X}[:, 0, t] \in \mathbb{R}^{N \times 1 \times 1}$.

However, depending on the application domain, using a transformed version of the time series to cluster members can be preferable. Similarly, considering a time window around each given time step can alleviate issues when working with highly fluctuating time series. Finally, DTW is a very well established tool in time series analysis that shifts the focus towards the global shapes of the times series rather than their differences at each time step. Therefore, `PersiGraph` has now been extended to handle these cases.

From now on, we will refer to the provided data as the *original* data $\mathbf{X_t}$ and the transformed data as the *transformed* data $\mathbf{X}_{t, trans}$. In addition, we distinguish between the data that is used to cluster members and compute clustering scores $\mathbf{X}_{t, clus}$, which is based on $\mathbf{X}_{t, trans}$ and the data that is used to define the graph components $\mathbf{X}_{t, comp}$, which is based on $\mathbf{X}_{t}$.

Note that $\mathbf{X}_{t, zero}$ can also be transformed into $\mathbf{X}_{t, zero, clus}$ similarly to $\mathbf{X}_{t, trans}$. Likewise, $\mathbf{X}_{t, zero, clus}$ and $\mathbf{X}_{t, zero, clus}$ are defined.

### Multivariate time series

XXX What are the fundamental changes? Not in terms of implementation but in terms of generalization / design choices? XXX

### Transformed ensemble

Transforming the original data $\mathbf{X_t}$ into $\mathbf{X}_{t, trans}$ can be useful when the distances between original datapoints in $\mathbf{X}_t$ don't separate points that correspond to drastically different behaviour in the application domain.

In weather and climate prediction, complex periodical phenomenons are sometimes represented by bivariate indices $\mathtt{ind_{1|2}}$, where $\mathtt{ind_{1|2}}$ values are centered in and revolve around $(0, 0)$. The unit circle, i.e., the circle centered in $(0, 0)$ and of radius 1 plays often a key role in such indices. When the index is inside this circle, the weather or climate phenomenon studied is considered inexistent or weak and the actual $\mathtt{ind_{1|2}}$ values are then considered irrelevant, and, when outside, the phenomenon is considered strong and studying actual $\mathtt{ind_{1|2}}$ values becomes of interest to the domain expert. This is the case for the RMM index that study the MJO.

In the MJO case for example, the radius $r = \sqrt{\mathtt{ind_{1}}^2 + \mathtt{ind_{2}}^2}$ plays en even more determinant part in the sense that high ($>1$) $r$ values correspond to extreme weather events, and the greater $r$, the more extreme the event. Clustering datapoints based on a squared transformation of the radius can then help separate predictions describing a non existent MJO from an extreme event scenario. More specifically, if $\mathtt{ind_{1|2}}$ values are stored in $\mathbf{X}_t$, then  $\mathbf{X}_{t, trans}$ can be derived using $\mathtt{ind_{1|2}}_{trans} = r \times \mathtt{ind_{1|2}}$

The original data $\mathbf{X_t}$ is still useful to define the vertices and edges that will represent these clusters, as we will see later.

### Time window

#### Notations

Given an array $\mathtt{arr}$ of length $T$, we are considering consecutive subsets of this array, each called $\mathtt{window}$, using a sliding window mechanism of maximal length $w$ around each element $\mathbf{x} \in \mathtt{arr}$. We denote $\mathtt{mid_w} \in [0, \cdots, w-1]$ (*window midpoint*) the index in $\mathtt{window}$ corresponding to $\mathbf{x}$. The stride of the sliding window is set to $1$.

Towards the start and end boundaries of the array, the extracted window are shorter (implicit padding), we denote $w_t$ the actual length of the $t^{th}$ window extracted from $\mathtt{arr}$. In addition, towards the start (respectively end) boundaries, $\mathtt{mid_w}$ is shifted towards the left (resp. right) of $\mathtt{window}$. Start boundary conditions correspond to indices $\mathtt{bound}_{start} = 0, \cdots, \left \lfloor \frac{w-1}{2} \right \rfloor -1$ and end boundary conditions to $\mathtt{bound}_{end} = \left \lfloor \frac{w}{2} \right \rfloor, \cdots,  T-1$.

#### Indices used to extract windows

Outside boundary conditions:

- $[t - \left \lfloor \frac{w-1}{2} \right \rfloor, \cdots, t, \cdots, t + \frac{w}{2}]$ when $w$ is even.
- $[t - \frac{w-1}{2}, \cdots, t, \cdots, t + \frac{w-1}{2}]$ when $w$ is odd.

Inside boundary conditions:

- $[0, \cdots,  t + \left \lfloor \frac{w}{2} \right \rfloor], \; \forall t \in \mathtt{bound}_{start}$
- $[t - \left \lfloor \frac{w-1}{2} \right \rfloor, \cdots, T-1], \; \forall t \in \mathtt{bound}_{end}$

#### Midpoints

Outside boundary conditions:

- $\mathtt{mid_w} = \left \lfloor \frac{w-1}{2} \right \rfloor$, which implicitly defines the principle "center the window around $\mathbf{x}$ and favour future time steps when necessary".

Inside boundary conditions:

- $\mathtt{mid_w} = t \; \forall t \in \mathtt{bound}_{start}$
- $\mathtt{mid_w} = \left \lfloor \frac{w-1}{2} \right \rfloor \; \forall t \in \mathtt{bound}_{end}$. Note that this case is the same as the base case.

#### Use of sliding window in `PersiGraph`

Extracted windows are used to define $\mathbf{X}_{t, clus} \in \mathbb{R}^{N \times w_t \times d}$ from $\mathbf{X}_{t, trans} \in \mathbb{R}^{N \times d}$

Midpoints are used to define vertices and edges properties (expected values and uncertainty), as detailed in the "Vertices and Edges" section.

### DTW

XXX ? XXX

### Members zero

XXX Is there anything to say here as well? XXX

Cluster data
-------------------------------------------------------------------------------

The data that is used to cluster the members is defined is several steps. First transform the original data, then extract windows, and finally consider extracted windows as variables if DTW is not used.

### Transformed data

For each $t$ in $[0, \cdots, T-1]$, transform $\mathbf{X_t}$ into $\mathbf{X}_{t, trans}$, both in $ \in \mathbb{R}^{N \times d}$.

From now on, we referring to the data that should have gone through this transformation step, we will write $\mathbf{X}_{t, trans}$ regardless of whether the user actually wanted a transformation (we consider that the transformation used was then the identity function).

### Extract windows

Extract windows from $\mathbf{X}_{t, trans} \in \mathbb{R}^{N \times d}$ to define $\mathbf{X}_{t, clus} \in \mathbb{R}^{N \times w_t \times d}$ using the sliding window mechanism defined earlier. If DTW is not used, $\mathbf{X}_{t, clus}$ should be seen as a 2D matrix $\mathbf{X}_{t, clus} \in \mathbb{R}^{N \times (w_t \times d)}$

Note that if $w=1$, then it is equivalent to not using windows.

Clustering methods
-------------------------------------------------------------------------------

Any clustering method that has for main parameter $k$, the number of clusters, is suitable.

When using DTW, clustering methods have to be adapted to compute pairwise distances as well as cluster centers in a consistent way. In practice, `Persigraph` uses `tslearn` when using DTW, otherwise `sklearn`.

The chosen clustering method is applied for all $t \in [0, \cdots, T-1]$ and all $k \in [0, \cdots, N]$. (See the definition of "ensemble zero" for the assumption $k=0$).

Clustering scores
-------------------------------------------------------------------------------

There exist natural score functions to evaluate qualities of different clusterings (inertia, variance, diameter, log-likelihood etc.), but none of them is ideal, especially with few datapoints and when the case $k=1$ has to be considered as well.

Some score functions can be applied individually to each cluster (each being a subset of $\mathbf{X}_{t, clus}$) and then combined to give the score of the whole clustering (a partition of $\mathbf{X}_{t, clus}$). For example, compute the inertia of all clusters individually, and then the sum/mean/max of the inertia of all clusters yields the score of the clustering. Alternatively, log-likelihood evaluate the entire clustering at once.

In the context of `PersiGraph`:

- A score doesn't have to be monotonous with respect to $k$ the number of clusters, but empirical results seem better with the ones that are monotonous (e.g. max/mean of cluster variance), or "almost monotonous" (e.g. sum of cluster variance)
- A score doesn't have to be positive (log-likelihood can be negative for example)
- If the clustering score doesn't distinguish between the case $k=0$ and $k=1$, another $k>0$ assumption will be used to calibrate the worst score and therefore will never be considered as a relevant option. This can be problematic when the score is monotonous, one $k$ value (typically $k=1$) will never be considered.

We denote $r_{t,}$

### DTW

When using DTW, the concept of barycenter (DBA) is used instead of the usual definition of the mean as the cluster center for scores that require it. Similarly, scores that require pairwise distances are also computed by finding the DTW path first.

#### Option 1

Compute pairwise distances on the entire window using DTW path

#### Option 2

Compute pairwise distances between the midpoint of the barycenter and the corresponding time step for each member in the cluster.

Note that several time steps can correspond to the midpoint of the barycenter.

#### Option 3

Take a smaller time window around the midpoint to compute the pairwise distance after aligning the larger windows.

Clustering ratios and life spans
-------------------------------------------------------------------------------

For each time step, PersiGraph compute the clustering scores for all assumptions $k=0, ..., N$.

Sum of life spans is 1.

Algorithm steps
-------------------------------------------------------------------------------


Vertices and Edges
-------------------------------------------------------------------------------

#### Mean / barycenter

#### Standard deviation

