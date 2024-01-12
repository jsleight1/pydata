# Exploratory data analysis using pydata

## `pydata`

`pydata` is a python package developed for the analysis and
visualisation of L-shaped datasets. It can either be installed locally
using `poetry` or an docker container environment has also be created
which can be used for development and running the following examples.

`docker run -it docker.io/jsleight1/pydata:3.10-latest`

## Pydata generation and handling

``` python
import matplotlib.pyplot as plt
from pydata.pydata import pydata

x=pydata.example_pydata(type="iris")
print(x)
```

    pydata object:
     - Dimensions: 150 (samples) x 4 (features)

``` python
print(x.data)
```

                  Sample1  Sample2  Sample3  Sample4  Sample5  Sample6  Sample7  \
    sepal_length      5.1      4.9      4.7      4.6      5.0      5.4      4.6   
    sepal_width       3.5      3.0      3.2      3.1      3.6      3.9      3.4   
    petal_length      1.4      1.4      1.3      1.5      1.4      1.7      1.4   
    petal_width       0.2      0.2      0.2      0.2      0.2      0.4      0.3   

                  Sample8  Sample9  Sample10  ...  Sample141  Sample142  \
    sepal_length      5.0      4.4       4.9  ...        6.7        6.9   
    sepal_width       3.4      2.9       3.1  ...        3.1        3.1   
    petal_length      1.5      1.4       1.5  ...        5.6        5.1   
    petal_width       0.2      0.2       0.1  ...        2.4        2.3   

                  Sample143  Sample144  Sample145  Sample146  Sample147  \
    sepal_length        5.8        6.8        6.7        6.7        6.3   
    sepal_width         2.7        3.2        3.3        3.0        2.5   
    petal_length        5.1        5.9        5.7        5.2        5.0   
    petal_width         1.9        2.3        2.5        2.3        1.9   

                  Sample148  Sample149  Sample150  
    sepal_length        6.5        6.2        5.9  
    sepal_width         3.0        3.4        3.0  
    petal_length        5.2        5.4        5.1  
    petal_width         2.0        2.3        1.8  

    [4 rows x 150 columns]

``` python
print(x.description.head(2))
```

            ID Species
    0  Sample1  setosa
    1  Sample2  setosa

``` python
print(x.annotation)
```

                 ID    type
    0  sepal_length  length
    1   sepal_width   width
    2  petal_length  length
    3   petal_width   width

``` python
s=x.subset(
    samples=["Sample1", "Sample30", "Sample52"], 
    features=["sepal_length", "petal_length"]
)
print(s)
```

    pydata object:
     - Dimensions: 3 (samples) x 2 (features)

``` python
t=x.transpose()
print(t)
```

    pydata object:
     - Dimensions: 4 (samples) x 150 (features)

## Dimension reduction

### PCA

``` python
x.perform_dimension_reduction("pca")

print(x.pcs)
```

    pca object:
     - Dimensions: 150 (samples) x 2 (pca components)
     - Scaling: Zscore
     - Method: SVD

``` python
print(x.pcs.data)
```

           Sample1   Sample2   Sample3   Sample4   Sample5   Sample6   Sample7  \
    PCA1 -2.264703 -2.080961 -2.364229 -2.299384 -2.389842 -2.075631 -2.444029   
    PCA2  0.480027 -0.674134 -0.341908 -0.597395  0.646835  1.489178  0.047644   

           Sample8   Sample9  Sample10  ...  Sample141  Sample142  Sample143  \
    PCA1 -2.232847 -2.334640 -2.184328  ...   2.014810   1.901784   1.157882   
    PCA2  0.223148 -1.115328 -0.469014  ...   0.613886   0.689575  -0.698870   

          Sample144  Sample145  Sample146  Sample147  Sample148  Sample149  \
    PCA1   2.040558   1.998147   1.870503   1.564580   1.521170   1.372788   
    PCA2   0.867521   1.049169   0.386966  -0.896687   0.269069   1.011254   

          Sample150  
    PCA1   0.960656  
    PCA2  -0.024332  

    [2 rows x 150 columns]

``` python
print(x.pcs.description)
```

                ID    Species
    0      Sample1     setosa
    1      Sample2     setosa
    2      Sample3     setosa
    3      Sample4     setosa
    4      Sample5     setosa
    ..         ...        ...
    145  Sample146  virginica
    146  Sample147  virginica
    147  Sample148  virginica
    148  Sample149  virginica
    149  Sample150  virginica

    [150 rows x 2 columns]

``` python
print(x.pcs.annotation)
```

         ID  Percentage variance explained
    0  PCA1                      72.962445
    1  PCA2                      22.850762

### LDA

``` python
x.perform_dimension_reduction("lda", target="Species")
```

``` python
print(x.lda)
```

    lda object:
     - Dimensions: 150 (samples) x 2 (lda components)
     - Scaling: Zscore
     - Target: Species

``` python
print(x.lda.data)
```

           Sample1   Sample2   Sample3   Sample4   Sample5   Sample6   Sample7  \
    LDA1  8.061800  7.128688  7.489828  6.813201  8.132309  7.701947  7.212618   
    LDA2 -0.300421  0.786660  0.265384  0.670631 -0.514463 -1.461721 -0.355836   

           Sample8   Sample9  Sample10  ...  Sample141  Sample142  Sample143  \
    LDA1  7.605294  6.560552  7.343060  ...  -6.653087  -5.105559  -5.507480   
    LDA2  0.011634  1.015164  0.947319  ...  -1.805320  -1.992182   0.035814   

          Sample144  Sample145  Sample146  Sample147  Sample148  Sample149  \
    LDA1  -6.796019  -6.847359  -5.645003  -5.179565  -4.967741  -5.886145   
    LDA2  -1.460687  -2.428951  -1.677717   0.363475  -0.821141  -2.345091   

          Sample150  
    LDA1  -4.683154  
    LDA2  -0.332034  

    [2 rows x 150 columns]

### t-SNE

``` python
x.perform_dimension_reduction("tsne")
```

``` python
print(x.tsne)
```

    tsne object:
     - Dimensions: 150 (samples) x 2 (tsne components)
     - Scaling: Zscore

``` python
print(x.tsne.data)
```

             Sample1    Sample2    Sample3    Sample4    Sample5    Sample6  \
    TSNE1 -26.283514 -22.458551 -23.502516 -22.724354 -26.786913 -28.650599   
    TSNE2  -0.934040  -1.582246  -0.568364  -0.502646  -0.364475  -0.741429   

             Sample7    Sample8    Sample9   Sample10  ...  Sample141  Sample142  \
    TSNE1 -24.550871 -25.286180 -21.671492 -23.178801  ...  12.926679  12.963368   
    TSNE2   0.185788  -0.886462  -0.708517  -1.504229  ...   0.023204   0.624381   

           Sample143  Sample144  Sample145  Sample146  Sample147  Sample148  \
    TSNE1   7.881903  13.546475  13.831253   12.38452   6.782580  11.200541   
    TSNE2  -1.946566   0.282336  -0.369666    0.20085  -2.073794   0.227081   

           Sample149  Sample150  
    TSNE1  13.349828   8.691272  
    TSNE2  -1.299510   0.213793  

    [2 rows x 150 columns]

### UMAP

``` python
x.perform_dimension_reduction("umap")
```

    /usr/local/lib/python3.10/site-packages/umap/umap_.py:1943: UserWarning:

    n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.

``` python
print(x.umap)
```

    umap object:
     - Dimensions: 150 (samples) x 2 (umap components)
     - Scaling: Zscore

``` python
print(x.umap.data)
```

             Sample1    Sample2    Sample3    Sample4    Sample5    Sample6  \
    UMAP1  13.386802  13.435769  14.054417  13.881824  13.202879  13.335918   
    UMAP2   9.793719   7.594136   7.752531   7.409498  10.024071  11.176532   

             Sample7    Sample8    Sample9   Sample10  ...  Sample141  Sample142  \
    UMAP1  13.591583  13.632071  13.486156  13.657273  ...   3.845543   4.033737   
    UMAP2   8.710538   9.101402   7.161908   7.695811  ...   1.539070   1.608526   

           Sample143  Sample144  Sample145  Sample146  Sample147  Sample148  \
    UMAP1   3.059679   3.746723   3.564396   3.981827   3.312455   4.161446   
    UMAP2  -0.596922   1.913777   1.807070   1.468371  -0.912279   0.848303   

           Sample149  Sample150  
    UMAP1   3.310499   4.284542  
    UMAP2   1.561345  -0.664872  

    [2 rows x 150 columns]

## Plotting

``` python
x.plot(type="pca", colour_by="Species")
```

![](README_files/figure-commonmark/cell-21-output-1.png)

``` python
x.plot(type="lda")
```

![](README_files/figure-commonmark/cell-22-output-1.png)

``` python
x.plot(type="tsne", colour_by="Species")
```

![](README_files/figure-commonmark/cell-23-output-1.png)

``` python
x.plot(type="umap", colour_by="Species")
```

![](README_files/figure-commonmark/cell-24-output-1.png)

``` python
x.transpose().plot(type="violin", fill=False)
```

![](README_files/figure-commonmark/cell-25-output-1.png)

``` python
x.plot(
    type="feature_heatmap", 
    annotate_samples_by=["Species"], 
    annotate_features_by=["type"], 
    xticklabels=False
)
```

![](README_files/figure-commonmark/cell-26-output-1.png)

``` python
x.plot(
    type="correlation_heatmap", 
    annotate_samples_by=["Species"], 
    xticklabels=False,
    yticklabels=False
)
```

    /usr/local/lib/python3.10/site-packages/seaborn/matrix.py:560: UserWarning:

    Clustering large matrix with scipy. Installing `fastcluster` may give better performance.

    /usr/local/lib/python3.10/site-packages/seaborn/matrix.py:560: UserWarning:

    Clustering large matrix with scipy. Installing `fastcluster` may give better performance.

![](README_files/figure-commonmark/cell-27-output-2.png)

``` python
x.plot(type="distribution", kind="kde", legend=False)
```

![](README_files/figure-commonmark/cell-28-output-1.png)

``` python
x.transpose().subset(["sepal_length"]).plot(type="distribution", kde=True)
```

![](README_files/figure-commonmark/cell-29-output-1.png)

``` python
x.transpose().plot(type="distribution", kind="ecdf")
```

![](README_files/figure-commonmark/cell-30-output-1.png)

``` python
x.transpose().plot(type="scatter", xaxis="petal_length", yaxis="sepal_length")
```

![](README_files/figure-commonmark/cell-31-output-1.png)
