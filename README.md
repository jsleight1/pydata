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
x.data
x.description
x.annotation
```

    pydata object:
     - Dimensions: 5 (samples) x 20 (features)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | ID        |
|-----|-----------|
| 0   | Feature1  |
| 1   | Feature2  |
| 2   | Feature3  |
| 3   | Feature4  |
| 4   | Feature5  |
| 5   | Feature6  |
| 6   | Feature7  |
| 7   | Feature8  |
| 8   | Feature9  |
| 9   | Feature10 |
| 10  | Feature11 |
| 11  | Feature12 |
| 12  | Feature13 |
| 13  | Feature14 |
| 14  | Feature15 |
| 15  | Feature16 |
| 16  | Feature17 |
| 17  | Feature18 |
| 18  | Feature19 |
| 19  | Feature20 |

</div>

## Linear dimensionality reduction using principal component analysis (PCA).

``` python
x.compute_pca()

print(x.pcs)

x.pcs.data
x.pcs.description
x.pcs.annotation
```

    pca object:
     - Dimensions: 5 (samples) x 5 (principal components)
     - Scaling: Zscore
     - Method: SVD

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | ID  | Percentage variance explained |
|-----|-----|-------------------------------|
| 0   | PC1 | 38.5524                       |
| 1   | PC2 | 24.005699                     |
| 2   | PC3 | 20.933268                     |
| 3   | PC4 | 16.508632                     |
| 4   | PC5 | 0.0                           |

</div>

## Plotting

``` python
# PCA 
x.plot(type = "pca")
```

![](README_files/figure-commonmark/cell-4-output-1.png)
