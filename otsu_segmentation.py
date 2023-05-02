import numpy as np
import matplotlib.pyplot as plt
import cv2


def otsu(img):
    img = 255*(img/np.max(img))
    img = img.astype(np.uint8)

    sigma = np.zeros(256)

    for i in np.arange(0,256,1):
        # Thresholding the image using grayscale value i
        binarized_image = img>i
        # Computing conditional statistics
        std_foreground = np.nanstd(img[binarized_image])
        std_background = np.nanstd(img[1-binarized_image])
        cdf_foreground = np.sum(binarized_image)/np.size(img)
        cdf_background = 1-cdf_foreground
        # Computing the intra-class variance
        sigma[i] =  np.sqrt(cdf_background*std_background**2 + cdf_foreground*std_foreground**2)

    otsu_threshold = int(np.nanargmin(sigma))

    return otsu_threshold