import jax.numpy as jnp
import jax.linalg as jla
from jax import jit, vmap
from ..data import dataset_shapes, flatten_features   

def CCA(data1, data2):
    """
    Canonical Correlation Analysis. 
    The compressed data dimension is the minimum of the dimensions of data1 and data2.

    Parameters
    ----------
    data1: jax.Array
        Data array of shape (n_samples, feature_shape1) e.g. parameters.
    data2: jax.Array
        Paired data array data1 of shape (n_samples, feature_shape2) e.g. simulations.

    Returns
    -------
    tuple of jax.Array
        A 3-tuple ``(compressed_data, compressed_params, canonical_correlations)`` with:
        - `compressed_data`: The compressed data array of shape (n_samples, min(feature_shape1, feature_shape2)).
        - `compressed_params`: The compressed parameters array of shape (n_samples, min(feature_shape1, feature_shape2)).
        - `canonical_correlations`: The canonical correlations of shape (min(feature_shape1, feature_shape2),).

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
    if n_features1 > n_features2:
        raise ValueError("The flattened dimension of data1 is greater than the flattened dimension of data2. The higher dimensional data should be the data1.")
    
    # Flatten the features if 2D
    data1, data2 = flatten_features(data1), flatten_features(data2)
    n_features1, n_features2 = dataset_shapes(data1)[1], dataset_shapes(data2)[1]

    # Rename the variables 
    # The higher dimensional data would be the 'data' vector 't'
    # The lower dimensional data would be the 'parameter' vector 'p'
    if n_features1 > n_features2:
        data1, data2 = data_t, data_p
        t_dim, p_dim = n_features1, n_features2
        print("'data1' is the higher dimensional data that will be compressed.")
    elif n_features1 == n_features2:
        data1, data2 = data_p, data_t
        t_dim, p_dim = n_features1, n_features2
        print("The flattened dimensions of 'data1' and 'data2' are the same. We will compress 'data1'.")
    else:
        data1, data2 = data_p, data_t
        t_dim, p_dim = n_features1, n_features2
        print("'data2' is the higher dimensional data that will be compressed.")

    # Implemented following the method in https://github.com/98minsu/CosmoCompression/blob/main/notebook.ipynb
    # cov_cca is composed of the covariance matrices of data_p and data_t, and the cross-covariance matrix between data_p and data_t.
    # [[ cov_p    , cov_pt ],
    #  [ cov_pt.T , cov_t  ]]
    cov_cca = jnp.cov(data_p.T, data_t.T)
    cov_p = cov_cca[:p_dim, :p_dim]
    cov_t = cov_cca[p_dim:, p_dim:]
    cov_pt = cov_cca[:p_dim, p_dim:]
    cov_l = cov_pt.T @ jnp.linalg.inv(cov_p) @ cov_pt # cov_tp is the transpose of cov_pt by symmetry of the covariance matrix
    evals, evecs = jnp.linalg.eigh(cov_t, cov_t-cov_l) # use .eigh since the covariance matrices are symmetric

    # In the context of the CCA, only min( dim(param), dim(data vector) ) components are real and the rest are noise. 
    evals = evals[::-1][:n_params]
    evecs = evecs[:,::-1][:,:n_params]

    # Transform data vector into the canonical space
    canonical_tmatrix = evecs
    compressed_data = data_t @ evecs

    # Transform the parameters into the canonical space
    canon_corr = jnp.sqrt(evals/(1+evals))
    canonical_pmatrix = jnp.linalg.inv(cov_p) @ cov_pt @ jnp.diag(1/jnp.sqrt(evals))
    compressed_params = data_p @ canonical_pmatrix

    return compressed_data, compressed_params, canon_corr



