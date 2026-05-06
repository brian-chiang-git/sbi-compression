import jax.numpy as jnp
import jax 
from jax.dtypes

def dataset_shape(dataset) -> tuple(int, tuple):
    """
    Dataset shape utility function.

    Parameters
    ----------
    dataset : array-like
        Input array of shape (n_samples, feature_shape).

    Returns
    -------
    tuple of jax.Array
        A 2-tuple ``(n_samples, feature_shape)`` with:
        - ``n_samples`` of shape ``()``.
        - ``feature_shape``.

    Raises
    ------
    ValueError
        If len(dataset.shape) < 2. Dataset should have at least 2 dimensions, i.e. there should be data and parameter pairs.

    Examples
    --------
    >>> dataset_shape(dataset)
    """
    if not isinstance(dataset, jax.Array):
        dataset = jnp.asarray(dataset)
    if len(dataset.shape) < 2:
        raise ValueError("Input dataset has less than 2 dimensions, i.e. there is no data and parameter pair.")
    n_samples = dataset.shape[0]
    feature_shape = dataset.shape[1:]
    return n_samples, feature_shape

def flatten_features(dataset):
    """
    Flattens the features of a dataset if it is 2D, otherwise returns the data as is.

    Parameters
    ----------
    dataset : array-like
        Input array of shape (n_samples, feature_shape).

    Returns
    -------
    flattened_dataset : array-like
        Flattened dataset of shape (n_samples, -1) if input is 2D, otherwise same as input.

    Examples
    --------
    >>> flatten_features(dataset)
    """
    if not isinstance(dataset, jax.Array):
        dataset = jnp.asarray(dataset)
    if len(dataset.shape) == 2:
        return dataset
    elif len(dataset.shape) == 3:
        n_samples = dataset.shape[0]
        return dataset.reshape(n_samples, -1)
    else:
        raise ValueError("Input data must be 1D or 2D array.")
