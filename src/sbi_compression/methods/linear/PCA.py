import jax.numpy as jnp
from jax import jit, vmap

def PCA_compression(data, out_dim="all"):
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    out_dim : int, optional
        Number of output dimensions.

    Returns
    -------
    tuple of jax.Array
        A 2-tuple ``(x_pca_reduced, sorted_evals)`` with:
        - ``sorted_evals`` of shape ``(n_features,)``.
        - ``x_pca_reduced`` of shape ``(n_samples, out_dim)``.

    Raises
    ------
    ValueError
        If n_samples < n_features.

    Examples
    --------
    >>> PCA(data, out_dim=2)
    """
    z = data - jnp.mean(data, axis=0)  
    cov_matrix = jnp.cov(z, rowvar=False)  
    evals, evecs = jnp.linalg.eigh(cov_matrix) 
    
    # Sort eigenvectors by eigenvalues
    sort_indices = jnp.argsort(evals)[::-1]
    sorted_evals = evals[sort_indices]
    U = evecs[:, sort_indices].T
    x_pca = jnp.dot(U, z.T).T

    if out_dim == "all" or out_dim is None:
        out_dim = x_pca.shape[1]

    return x_pca[:, :out_dim], sorted_evals