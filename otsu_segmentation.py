import numpy as np
import matplotlib.pyplot as plt
import cv2


def otsu(img):
    img = 255*(img/np.max(img))
    img = img.astype(np.uint8)

    var = np.zeros(256)
    
    # Using the inter-class variance (first statistical moments only)
    for i in np.arange(1,255,1):
        # Thresholding the image using grayscale value i
        binarized_image = img>i
        # Computing conditional statistics
        cdf_foreground = np.sum(binarized_image)/np.size(img)
        cdf_background = 1-cdf_foreground
        mean_foreground = (np.mean(img[binarized_image == 1]))
        mean_background = (np.mean(img[binarized_image == 0]))
        # Computing the intra-class variance
        var[i] =  cdf_background*cdf_foreground*(mean_background-mean_foreground)**2
    
    
    # Determining the Otsu threshold (the last element is discarded due to empty foreground array)
    otsu_threshold = int(np.nanargmax(var[0:-2]))

    return otsu_threshold
