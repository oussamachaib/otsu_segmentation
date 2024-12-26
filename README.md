# otsu_segmentation

 Python implementation (coded entirely in numpy) of  ["A Threshold Selection Method from Gray-Level Histograms"](https://ieeexplore.ieee.org/document/4310076) (1979) -- Otsu's method for autonomous image segmentation.
 
 The algorithm takes a grayscale image as input, and outputs the grayscale threshold maximizing the interclass variance (or equivalently, minimizing the intraclass variance). 
 
 In the Jupyter Notebook, I compare my personal implementation to OpenCV's. Moreover, I show how Otsu's method is nothing but a special case of the k-means clustering algorithm proposed by [MacQueen](https://books.google.co.uk/books?hl=en&lr=&id=IC4Ku_7dBFUC&oi=fnd&pg=PA281&dq=related:NfMlILJJH88J:scholar.google.com/&ots=nQVgC-ObtP&sig=Ci0RxETPxwKxHViD3j5GXDezdlU&redir_esc=y#v=onepage&q&f=false) in 1967 (more than 10 years earlier!)

##### Contents
---------------
```otsu_segmentation.py```: Python implementation of the algorithm.

```demo.ipynb```: A demonstration and comparison against two state-of-the-art methods.