# Exploratory data analysis using pydata

## `pydata`

`pydata` is a python package developed for the analysis and
visualisation of L-shaped datasets. It can either be installed locally
using `poetry` or an docker container environment has also be created
which can be used for development and running the following examples.

`docker run -it docker.io/jsleight1/pydata:3.10-latest`

``` python
from pydata.pydata import pydata

x = pydata.example_pydata()

print(x)
print(x.data.head(2))
print(x.description.head(2))
print(x.annotation.head(2))
```

    pydata object:
     - Dimensions: 6 (samples) x 20 (features)
              Sample1  Sample2  Sample3  Sample4  Sample5  Sample6
    Feature1        1        5        3        4        2        1
    Feature2        7        9        0        1        4        3
            ID Treatment
    0  Sample1   Control
    1  Sample2   Control
             ID
    0  Feature1
    1  Feature2

## Linear dimensionality reduction using principal component analysis (PCA).

``` python
x.compute_pca()

print(x.pcs)

print(x.pcs.data)
print(x.pcs.description.head(2))
print(x.pcs.annotation)
```

    pca object:
     - Dimensions: 6 (samples) x 5 (principal components)
     - Scaling: Zscore
     - Method: SVD
          Sample1   Sample2   Sample3   Sample4   Sample5   Sample6
    PC1  0.015036 -1.685930 -1.969292  3.881869 -3.136891  2.895207
    PC2 -3.037243 -1.546258  4.576623  1.268257 -0.858779 -0.402600
    PC3 -1.994504  1.960133 -1.392191  2.622691  1.015385 -2.211514
    PC4 -1.855321  2.221158 -0.143624 -1.218673 -0.884526  1.880987
    PC5 -1.111968 -1.199684 -0.649938 -0.105593  1.948909  1.118274
            ID Treatment
    0  Sample1   Control
    1  Sample2   Control
        ID Percentage variance explained
    0  PC1                     33.343255
    1  PC2                      29.22442
    2  PC3                     18.798897
    3  PC4                      11.83504
    4  PC5                      6.798388

## Plotting

``` python
# PCA 
x.plot(type = "pca", colour_by = "Treatment")
```

![](README_files/figure-commonmark/cell-4-output-1.png)
