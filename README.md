# otsu_segmentation

 Python implementation of  ["A Threshold Selection Method from Gray-Level Histograms"](https://ieeexplore.ieee.org/document/4310076) (1979) -- Otsu's method for autonomous image segmentation.
 
 The algorithm takes a grayscale image as input, and outputs the grayscale threshold maximizing the interclass variance (or equivalently, minimizing the intraclass variance).