from pydata.ldata import ldata
from pydata.pca import pca
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class pydata(ldata):
    """
    Simple dataset computation and visualisation.

    pydata objects use a L-shaped data structure where the underlying 
    data set is a DataFrame of numeric values with an associated 
    description and annotation DataFrame containing relevant sample 
    and feature metadata respectively.

    Examples
    --------

    >>> # Generate some random data
    >>> np.random.seed(38)
    >>> data=pd.DataFrame(
    >>>     np.random.randint(0, 10, size=100).reshape(20, 5),
    >>>     index=["Feature" + str(i)for i in range(1, 21)], 
    >>>     columns=["Sample" + str(i)for i in range(1, 6)] 
    >>> )
    >>>
    >>> # Generate sample metadata
    >>> desc=pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
    >>>
    >>> # Generate feature metadata
    >>> annot=pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})
    >>>
    >>> # Generate pydata object
    >>> x=pydata(data, desc, annot)
    >>> x
    >>> x.data
    >>> x.description
    >>> x.annotation
    """
    def __init__(self, data, description, annotation, pcs: pca=None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pd.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        annotation : pd.DataFrame
            A DataFrame of feature annotation with ID column matching 
            row names of data attribute.
        """

        super().__init__(data, description, annotation)
        self._pcs=pcs

    def __str__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def __repr__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def _get_pcs(self):
        return getattr(self, "_pcs")
    def _set_pcs(self, value: pca):
        if value is not None:
            assert isinstance(value, pca)
            value._validate()
        self._pcs=value
    pcs=property(_get_pcs, _set_pcs)

    def plot(self, type: str, **kwargs):
        match type:
            case "pca":
                self._pca_plot(**kwargs)
            case _:
                raise Exception(type + " plot type not implement")

    def _pca_plot(
            self, 
            xaxis: str="PC1", 
            yaxis: str="PC2", 
            colour_by: str="ID", 
            **kwargs
        ):
        if not isinstance(self.pcs, pca):
            self.computePCA(**kwargs)
        df=self.pcs.data.transpose()
        plt.scatter(df[xaxis], df[yaxis])
        plt.show()

    def computePCA(self, npcs: int=5, scaling: str="Zscore", method: str="SVD"):
        
        dat=self.data.transpose()

        match scaling: 
            case "none":
                dat=dat
            case "Zscore":
                dat=StandardScaler().fit_transform(dat)
            case _:
                raise Exception(scaling + " scaling method not implemented")

        match method:
            case "SVD":
                pcs=self._svd_pca(x=dat, npcs=npcs)
            case _:
                raise Exception(method + " pca method not implemented")
       
        pcs.scaling=scaling
        pcs.method=method
        self.pcs=pcs

    def _svd_pca(self, x, npcs):
        p=PCA(n_components=npcs)
        principalComponents=p.fit_transform(x)
        principalDf=pd.DataFrame(
            data=principalComponents, 
            columns=["PC" + str(i) for i in range(1, npcs+1)]
        )
        principalDf.index=self.description["ID"].tolist()
        varianceExplained=pd.DataFrame(
            [
                principalDf.columns.tolist(), 
                (p.explained_variance_ratio_ * 100).tolist()
            ]
        ).transpose()
        varianceExplained.columns=["ID", "Percentage variance explained"]
        out=pca(
            data=principalDf.transpose(), 
            description=self.description, 
            annotation=varianceExplained
        )
        return out
