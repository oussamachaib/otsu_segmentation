import numpy as np

class otsu:
    """
    Otsu segmentation algorithm.

    Computes the optimal binarization threshold (an integer between 0 and 255) from a 2D array.
    The returned integer is the threshold maximizing (minimizing) the inter-class (intra-class) variance.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    _______
    otsu_threshold: int
        Otsu threshold

    """
    def __init__(self):
        self.threshold = None
        self.var = None
        self.scan = None

    def _rescale(self, X):
        # Rescaling input arrays to an np.uint8 single-channel array with values between 0 and 2^8 (= 256)
        if X.dtype == 'nd.array':
            return (255 * X/X.max()).astype(np.uint8)
        else:
            return X

    def _sigma(self, X):
        # Computing the argmax of the inter-class variance
        # Note: This is equivalent to the argmin of the intra-class variance in the binary context but is more efficient given it only needs first moments (means)

        # One-dimensonalize image array
        # X = (X.flatten()).astype(float)

        # Initialize array of variances
        var_i = np.zeros(256)

        # Scanning full 8-bit range (0-255) for potential thresholds
        for i in np.arange(255):
            # Thresholding the image using grayscale value "i" -- strict inequality as per Otsu 1979
            isForeground = (X > i)

            # Class probabilities
            omega1 = np.sum(isForeground)/np.size(X)

            if omega1 == 0:
                mu_foreground = 0
                mu_background = X.mean()
            elif omega1 == 1:
                mu_foreground = X.mean()
                mu_background = 0
            else:
                mu_foreground = X[isForeground].mean()
                mu_background = X[isForeground == False].mean()

            # Computing the inter-class variance
            var_i[i] = (1-omega1) * (omega1) * (mu_background - mu_foreground) ** 2

        return var_i

    def fit(self, X):
        # Computing the Otsu threshold

        # Computing discriminant criterion for each grayscale value
        var_i = self._sigma(self._rescale(X))

        # Saving attributes
        self.scan = {'i': np.arange(256), 'var_i': var_i} # Individual thresholds
        self.var = np.max(var_i) # Intra-class variance
        self.threshold = np.argmax(var_i) # Optimal threshold

    def predict(self, X):
        # Computing the binary image
        return (X >= self.threshold).astype(int)










