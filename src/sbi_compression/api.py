
from .methods.linear.PCA import PCA_compression


def PCA(data, out_dim='all'):
    """
    Principal Component Analysis (PCA) API function.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    out_dim : int, optional
        Number of output dimensions.

    Returns
    -------
    numpy.ndarray
        Transformed array.

    Raises
    ------
    ValueError
        If X is empty.

    Examples
    --------
    >>> PCA(data, out_dim=2)
    """
    return PCA_compression(data, out_dim=out_dim)

def CCA(data1, data2):
    """Canonical Correlation Analysis (CCA) for finding relationships between two datasets."""
    # Placeholder for CCA implementation
    NotImplementedError("CCA function is not implemented yet.")

def MOPED(data1, data2, fiducial_data, forward_diff_data):
    """MOPED (Multi-Output Principal Component Analysis) for dimensionality reduction."""
    # Placeholder for MOPED implementation
    NotImplementedError("MOPED function is not implemented yet.")

def eMOPED(data1, data2):
    """eMOPED (extended MOPED) for enhanced dimensionality reduction."""
    # Placeholder for eMOPED implementation
    NotImplementedError("eMOPED function is not implemented yet.")

def AutoEncoder(data1, data2):
    """Autoencoder for nonlinear dimensionality reduction."""
    # Placeholder for AutoEncoder implementation
    NotImplementedError("AutoEncoder function is not implemented yet.")

def VMIM(data1, data2):
    """VMIM (Variational Mutual Information Maximization) for dimensionality reduction."""
    # Placeholder for VMIM implementation
    NotImplementedError("VMIM function is not implemented yet.")

def IMNN(data1, data2):
    """IMNN (Information Maximizing Neural Network) for dimensionality reduction."""
    # Placeholder for IMNN implementation
    NotImplementedError("IMNN function is not implemented yet.")