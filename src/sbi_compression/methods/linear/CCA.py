import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from ..data import dataset_shapes, flatten_features   
    
@jax.jit
def CCA(data1, data2, regularisation=0.0):
    """
    Canonical Correlation Analysis. 
    The compressed data dimension is the minimum of the dimensions of data1 and data2.

    Parameters
    ----------
    data1: jax.Array
        Data array of shape (n_samples, feature_shape1) e.g. parameters.
    data2: jax.Array
        Paired data array data1 of shape (n_samples, feature_shape2) e.g. simulations.
    regularisation: float
        Regularisation parameter to prevent the covariance matrices from being singular.

    Returns
    -------
    tuple of jax.Array
        A 5-tuple ``(compressed_data, compressed_params, canonical_correlations, canonical_t_directions, canonical_p_directions)`` with:
        - `compressed_data`: The compressed data array of shape (n_samples, min(feature_shape1, feature_shape2)).
        - `compressed_params`: The compressed parameters array of shape (n_samples, min(feature_shape1, feature_shape2)).
        - `canonical_correlations`: The canonical correlations of shape (min(feature_shape1, feature_shape2),).
        - `canonical_t_directions`: The canonical directions for the data array of shape (feature_shape2, min(feature_shape1, feature_shape2)).
        - `canonical_p_directions`: The canonical directions for the parameters array of shape (feature_shape1, min(feature_shape1, feature_shape2)).   
    Raises
    ------
    ValueError
        If data1 and data2 have different number of samples.

    Examples
    --------
    >>> CCA(data1, data2)
    """
    n_samples1, feature_shape1 = dataset_shapes(data1)
    n_samples2, feature_shape2 = dataset_shapes(data2)
    if n_samples1 != n_samples2:
        raise ValueError("The paired dataset (data1, data2) have different number of samples.")
    
    # Flatten the features if 2D
    data1, data2 = flatten_features(data1), flatten_features(data2)
    n_features1, n_features2 = dataset_shapes(data1)[1][0], dataset_shapes(data2)[1][0]
    if n_features1 > n_features2:
        raise ValueError("The flattened dimension of data1 is greater than the flattened dimension of data2. The higher dimensional data should be the data1.")

    # Rename the variables 
    # The higher dimensional data would be the 'data' vesctor 't'
    # The lower dimensional data would be the 'parameter' vector 'p'
    if n_features1 > n_features2:
        data_t, data_p = data1, data2
        t_dim, p_dim = n_features1, n_features2
        print("'data1' is the higher dimensional data that will be compressed.")
    elif n_features1 == n_features2:
        data_p, data_t = data1, data2
        p_dim, t_dim = n_features1, n_features2
        print("The flattened dimensions of 'data1' and 'data2' are the same. We will compress 'data1'.")
    else:
        data_p, data_t = data1, data2
        p_dim, t_dim = n_features1, n_features2
        print("'data2' is the higher dimensional data that will be compressed.")

    # Implemented following the method in https://github.com/98minsu/CosmoCompression/blob/main/notebook.ipynb
    # cov_cca is composed of the covariance matrices of data_p and data_t, and the cross-covariance matrix between data_p and data_t.
    # [[ cov_p    , cov_pt ],
    #  [ cov_pt.T , cov_t  ]]
    cov_cca = jnp.cov(data_p.T, data_t.T)
    cov_p = cov_cca[:p_dim, :p_dim] + regularisation * jnp.eye(p_dim)
    cov_t = cov_cca[p_dim:, p_dim:] + regularisation * jnp.eye(t_dim)
    cov_pt = cov_cca[:p_dim, p_dim:]
    # cov_l = cov_pt.T @ jla.inv(cov_p) @ cov_pt # cov_tp is the transpose of cov_pt by symmetry of the covariance matrix
    # evals, evecs = jsla.eigh(cov_t, cov_t-cov_l) # use .eigh since the covariance matrices are symmetric

    # Since the covariance matrices are symmetric and jax does not support generalized eigenvalue decomposition
    # We can use the Cholesky decomposition to solve the generalized eigenvalue problem.
    cov_l = cov_pt.T @ jla.inv(cov_p) @ cov_pt
    # B = cov_t - cov_l
    B = cov_t
    L = jnp.linalg.cholesky(B)
    Linv = jla.inv(L)
    # Transform to a standard eigenvalue problem: C = Linv @ A @ Linv.T
    # C = Linv @ cov_t @ Linv.T
    C = Linv @ cov_l @ Linv.T
    evals, evecs_standard = jla.eigh(C)
    # Back-transform eigenvectors: v = (L^T)^-1 @ evecs_standard
    evecs = jla.solve(L.T, evecs_standard)

    # In the context of the CCA, only min( dim(param), dim(data vector) ) components are real and the rest are noise. 
    evals = evals[::-1][:p_dim]
    evecs = evecs[:,::-1][:,:p_dim]

    # Transform data vector into the canonical space
    canonical_t_directions = evecs
    compressed_data = data_t @ evecs

    # Transform the parameters into the canonical space
    # canon_corr = jnp.sqrt(evals/(1+evals))
    canon_corr = jnp.sqrt(evals)
    canonical_p_directions = jla.inv(cov_p) @ cov_pt @ evecs @ jnp.diag(1/jnp.sqrt(evals))
    compressed_params = data_p @ canonical_p_directions

    return compressed_data, compressed_params, canon_corr, canonical_t_directions, canonical_p_directions



