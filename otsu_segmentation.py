import numpy as np
import matplotlib.pyplot as plt
import cv2


def otsu(img):
    """
    Otsu segmentation algorithm.

    Returns an integer between 0 and 255, corresponding to the optimal binarization threshold
    maximizing the inter-class variance between foreground and background pixels

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    _______
    otsu_threshold: int
        Otsu threshold

    """

    # Converting input arrays to an np.uint8 single-channel array
    X = 255*(img/np.max(img))
    X = X.astype(np.uint8)

    var = np.zeros(256)
    
    # Computing the inter-class variance (note this is easier than computing the intra-class variance
    # as it only requires first-order moments -- the final result should however be identical)
    for i in np.arange(256):
        # Thresholding the image using grayscale value i
        isForeground = X>=i
        # Conditional statistics
        P_foreground = np.sum(isForeground)/np.size(X)
        mu_foreground = 0
        mu_background = 0
        if sum(isForeground.ravel())>0: # Preventing operations on empty arrays
            mu_foreground = X[isForeground == 1].mean()
        if sum(isForeground.ravel())<np.size(X):  # Preventing operations on empty arrays
            mu_background = X[isForeground == 0].mean()
        # Computing the intra-class variance
        var[i] = P_foreground*(1-P_foreground)*(mu_background-mu_foreground)**2

    # Determining the Otsu threshold
    otsu_threshold = int(np.nanargmax(var))-1

    return otsu_threshold
