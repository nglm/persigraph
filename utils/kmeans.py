from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _validate_center_shape, _kmeans_single_lloyd
from sklearn.utils import check_array
import numpy as np


# REQUIRES v0.23.2 of sklearn!!

# See sklearn.utils.extmath.row_norms
def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.
    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    # if sparse.issparse(X):
    #     if not isinstance(X, sparse.csr_matrix):
    #         X = sparse.csr_matrix(X)
    #     norms = csr_row_norms(X)
    # else:
    norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

# See sklearn/utils/validation/check_random_state
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    # if isinstance(seed, numbers.Integral):
    #     return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class kmeans_custom(KMeans):

    def __init__(
        self, n_clusters=8, *, init='k-means++', n_init=10,
        max_iter=300, tol=1e-4, precompute_distances='deprecated',
        verbose=0, random_state=None, copy_x=True,
        n_jobs='deprecated', algorithm='auto'
        ):

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            precompute_distances=precompute_distances,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            n_jobs=n_jobs,
            algorithm=algorithm
            )


    def fit(self, X, y=None, sample_weight=None, x_squared_norms=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            .. versionadded:: 0.20
        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        self._check_params(X)
        random_state = check_random_state(self.random_state)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            _validate_center_shape(X, self.n_clusters, init)

        # # subtract of mean of x for more accurate distance computations
        # if not sp.issparse(X):
        #     X_mean = X.mean(axis=0)
        #     # The copy was already done above
        #     X -= X_mean

        #     if hasattr(init, '__array__'):
        #         init -= X_mean

        # precompute squared norms of data points
        if x_squared_norms is None:
            x_squared_norms = row_norms(X, squared=True)

        # if self._algorithm == "full":
        #     kmeans_single = _kmeans_single_lloyd
        # else:
        #     kmeans_single = _kmeans_single_elkan

        kmeans_single = _kmeans_single_lloyd

        best_labels, best_inertia, best_centers = None, None, None

        # seeds for the initializations of the kmeans runs.
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self._n_init)

        for seed in seeds:
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, self.n_clusters, max_iter=self.max_iter,
                init=init, verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, random_state=seed,
                n_threads=self._n_threads)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_

        # if not sp.issparse(X):
        #     if not self.copy_x:
        #         X += X_mean
        #     best_centers += X_mean

        # distinct_clusters = len(set(best_labels))
        # if distinct_clusters < self.n_clusters:
        #     warnings.warn(
        #         "Number of distinct clusters ({}) found smaller than "
        #         "n_clusters ({}). Possibly due to duplicate points "
        #         "in X.".format(distinct_clusters, self.n_clusters),
        #         ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None, x_squared_norms=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(
            X,
            sample_weight=sample_weight,
            x_squared_norms=x_squared_norms
            ).labels_