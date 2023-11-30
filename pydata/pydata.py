import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class pydata:
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
    >>> data = pd.DataFrame(
    >>>     np.random.randint(0, 10, size=100).reshape(20, 5),
    >>>     index=["Feature" + str(i)for i in range(1, 21)], 
    >>>     columns=["Sample" + str(i)for i in range(1, 6)] 
    >>> )
    >>>
    >>> # Generate sample metadata
    >>> desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
    >>>
    >>> # Generate feature metadata
    >>> annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})
    >>>
    >>> # Generate pydata object
    >>> x = pydata(data, desc, annot)
    >>> x
    >>> x.data
    >>> x.description
    >>> x.annotation
    """
    def __init__(self, data, description, annotation):
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

        self._data=data 
        self._description=description
        self._annotation=annotation
        self._validate()

    def __str__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def __repr__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    @property
    def data(self):
        return getattr(self, "_data")
    
    @data.setter
    def data(self, value: pd.DataFrame):
        """
        Set data attribute for pydata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        """
        assert isinstance(value, pd.DataFrame), "data is not DataFrame"
        self._check_dimnames(data=value)
        self._data=value

    @property
    def description(self):
        return getattr(self, "_description")
    
    @description.setter
    def description(self, value: pd.DataFrame):
        """
        Set description attribute for pydata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        """
        self._check_dimnames(description=value)
        self._description=value

    @property
    def annotation(self):
        return getattr(self, "_annotation")
    
    @annotation.setter
    def annotation(self, value: pd.DataFrame):
        """
        Set description attribute for pydata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of feature annotation with ID column matching 
            row names of data attribute.
        """
        self._check_dimnames(annotation=value)
        self._annotation=value

    @property
    def rownames(self):
        return self.data.index.values.tolist()

    @rownames.setter 
    def rownames(self, value: list):
        """
        Set feature names for pydata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert isinstance(value, list), "value must be list"
        assert len(value) == self.data.shape[0], "value does not match data dims"
        assert len(value) == len(set(value)), "value must contain unique IDs"
        self.annotation["ID"]=value
        self.data.index=value

    @property
    def colnames(self):
        return self.data.columns.tolist()
    
    @colnames.setter 
    def colnames(self, value: list):
        """
        Set sample names for pydata object.
        ------------------------------------
        value: list
            A list of sample names.
        """
        assert isinstance(value, list), "value must be list"
        assert len(value) == self.data.shape[1], "value does not match data dims"
        assert len(value) == len(set(value)), "value must contain unique IDs"
        self.description["ID"]=value
        self.data.columns=value

    @property 
    def dimnames(self):
        return [self.rownames, self.colnames]

    @dimnames.setter 
    def dimnames(self, value: list):
        assert isinstance(value, list), "value must be list"
        assert len(value) == 2, "value must be list with rownames and colnames"
        self.rownames=value[0]
        self.colnames=value[1]

    def _validate(self):
        assert isinstance(self.data, pd.DataFrame), "data is not DataFrame"
        assert isinstance(self.description, pd.DataFrame), \
            "description is not DataFrame"
        assert isinstance(self.annotation, pd.DataFrame), \
            "annotation is not DataFrame"
        self._check_dimnames()

    def _check_dimnames(
            self, 
            data: pd.DataFrame=None, 
            description: pd.DataFrame=None, 
            annotation: pd.DataFrame=None
        ):
        if data is None:
            data=self.data 
        if description is None:
            description=self.description
        if annotation is None:
            annotation=self.annotation
        rownames=data.index.values.tolist()
        colnames=data.columns.tolist()
        assert len(rownames) == len(set(rownames)), \
            "rownames must contain unique IDs"
        assert len(colnames) == len(set(colnames)), \
            "colnames must contain unique IDs"
        assert rownames == annotation["ID"].tolist(), \
            "data rownames do not match annotation ID"
        assert colnames == description["ID"].tolist(), \
            "data colnames do not match description ID"

    def plot(self, type: str, **kwargs):
        if type == "pca":
            self._pca_plot()
        else:
            raise Exception(type + " plot type not implement")


    def _pca_plot(self, colour_by: str="ID", **kwargs):
        df = self.pca(**kwargs)
        plt.scatter(df["PC1"], df["PC2"])

    def pca(self, npcs: int=5, scale: str="Zscore", method: str="SVD"):
        
        dat = self.data.transpose()

        if scale == "Zscore":
            dat = StandardScaler().fit_transform(dat)
        else: 
            raise Exception(scale + " scaling method not implement")

        if method == "SVD":
            out = self._svd_pca(x=dat, npcs=npcs)
        else:
            raise Exception(method + " pca method not implement")

        return out

    def _svd_pca(self, x, npcs):
        pca = PCA(n_components=npcs)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(
            data = principalComponents, 
            columns = ["PC" + str(i) for i in range(1, npcs+1)]
        )
        principalDf.index=self.description["ID"].tolist()
        return principalDf
