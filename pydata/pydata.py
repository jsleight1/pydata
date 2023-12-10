from pydata.ldata import ldata
from pydata.pca import pca
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

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
    >>> data = pd.DataFrame(
    >>>     np.random.randint(0, 10, size = 100).reshape(20, 5),
    >>>     index = ["Feature" + str(i)for i in range(1, 21)], 
    >>>     columns = ["Sample" + str(i)for i in range(1, 6)] 
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
    def __init__(self, data, description, annotation, pcs: pca = None):
        """
        Parameters
        ----------
        data: pd.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pd.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        annotation: pd.DataFrame
            A DataFrame of feature annotation with ID column matching 
            row names of data attribute.
        """

        super().__init__(data, description, annotation)
        self._pcs = pcs

    def __str__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def __repr__(self):
        return (
            f"pydata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    @staticmethod
    def example_pydata():
        np.random.seed(38)
        data = pd.DataFrame(
            np.random.randint(0, 10, size = 120).reshape(20, 6),
            index = ["Feature" + str(i)for i in range(1, 21)], 
            columns = ["Sample" + str(i)for i in range(1, 7)] 
        )
        grps = np.array(["Control", "Treatment"])
        desc = pd.DataFrame(
            {
                "ID": ["Sample" + str(i) for i in range(1, 7)], 
                "Treatment": np.repeat(grps, [3, 3], axis = 0)
            }
        )
        annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})
        return pydata(data, desc, annot)

    def _get_pcs(self):
        return getattr(self, "_pcs")
    def _set_pcs(self, value: pca):
        if value is not None:
            assert isinstance(value, pca)
            value._validate()
        self._pcs = value
    pcs = property(_get_pcs, _set_pcs)

    def plot(self, type: str, **kwargs):
        self._validate()
        match type:
            case "pca":
                self._pca_plot(**kwargs)
            case "violin":
                self._violin_plot(**kwargs)
            case "feature_heatmap": 
                self._feature_heatmap(**kwargs)
            case "correlation_heatmap":
                self._correlation_heatmap(**kwargs)
            case "density":
                self._density_plot(**kwargs)
            case _:
                raise Exception(type + " plot type not implement")

    def _plot_data(self):
        df = deepcopy(self.data)
        df["Feature"] = df.index 
        df = pd.melt(
            df, 
            id_vars = "Feature", 
            value_vars = self.colnames, 
            var_name = "Sample"
        )
        df = pd.merge(
            df, 
            self.description, 
            left_on = "Sample", 
            right_on = "ID"
        )
        df = df.drop(columns = ["ID"])
        df = pd.merge(
            df, 
            self.annotation, 
            left_on = "Feature", 
            right_on = "ID"
        )
        df = df.drop(columns = ["ID"])
        return df
        
    def _pca_plot(
            self, 
            xaxis: str = "PC1", 
            yaxis: str = "PC2", 
            colour_by: str = "ID", 
            **kwargs
        ):
        if not isinstance(self.pcs, pca):
            self.compute_pca(**kwargs)
        self.pcs._validate()
        df = deepcopy(self.pcs.data.transpose())
        df["ID"] = df.index 
        df = pd.merge(df, self.pcs.description, on = "ID")
        sns.relplot(data = df, x = xaxis, y = yaxis, hue = colour_by)

    def compute_pca(
            self, 
            npcs: int = 5, 
            scaling: str = "Zscore", 
            method: str = "SVD", 
            **kwargs
        ):
        """
        Parameters
        ----------
        npcs: Number of principal component to compute. Default is 5.
        scaling: Scaling method before PCA calculation. Default is "Zscore".
        method: PCA method for PCA calculation: Default is "SVD" for singular
            value decomposition.
        **kwargs: Passed to PCA method.
        """
        
        dat = self.data.transpose()

        match scaling: 
            case "none":
                dat = dat
            case "Zscore":
                dat = StandardScaler().fit_transform(dat)
            case _:
                raise Exception(scaling + " scaling method not implemented")

        match method:
            case "SVD":
                pcs = self._svd_pca(x = dat, npcs = npcs, **kwargs)
            case _:
                raise Exception(method + " pca method not implemented")
       
        pcs.scaling = scaling
        pcs.method = method
        self.pcs = pcs

    def _svd_pca(self, x, npcs, **kwargs):
        p = PCA(n_components = npcs, **kwargs)
        p_c = p.fit_transform(x)
        p_df = pd.DataFrame(
            data = p_c, 
            columns = ["PC" + str(i) for i in range(1, npcs+1)]
        )
        p_df.index = self.description["ID"].tolist()
        var_expl = pd.DataFrame(
            [
                p_df.columns.tolist(), 
                (p.explained_variance_ratio_ * 100).tolist()
            ]
        ).transpose()
        var_expl.columns = ["ID", "Percentage variance explained"]
        out = pca(
            data = p_df.transpose(), 
            description = self.description, 
            annotation = var_expl
        )
        return out

    def _violin_plot(self, colour_by: str = "Sample", **kwargs):
        sns.violinplot(
            data = self._plot_data(), 
            x = "Sample", 
            y = "value", 
            hue = colour_by
        )

    def _correlation_heatmap(
            self, 
            cor_method: str = "pearson", 
            annotate_samples_by = None,
            cmap = "coolwarm",
            **kwargs
        ):
        dat = self.data.corr(method = cor_method)
        if annotate_samples_by is not None:
            annotate_samples_by = self.colour_by_df(
                self.description, 
                annotate_samples_by
            )
            annotate_samples_by.index = self.description["ID"]

        self._heatmap(
            dat, 
            annotate_samples_by = annotate_samples_by, 
            cmap = cmap,
            cbar_kws = {"label": cor_method.capitalize() + " correlation"},
            **kwargs
        )


    def _feature_heatmap(
            self, 
            annotate_samples_by = None,
            annotate_features_by = None,
            cmap = "coolwarm",
            **kwargs
        ):
        if annotate_samples_by is not None:
            annotate_samples_by = self.colour_by_df(
                self.description, 
                annotate_samples_by
            )
            annotate_samples_by.index = self.description["ID"]

        if annotate_features_by is not None:
            annotate_features_by = self.colour_by_df(
                self.annotation, 
                annotate_features_by
            )
            annotate_features_by.index = self.annotation["ID"]

        self._heatmap(
            self.data, 
            cbar_kws = {"label": "Feature value"},
            annotate_samples_by = annotate_samples_by, 
            annotate_features_by = annotate_features_by, 
            cmap = cmap,
            **kwargs
        )

    @staticmethod
    def colour_by_df(x, colour_by):
        colours = []
        for i in colour_by:
            col = sns.color_palette("Spectral", n_colors = len(x[i].unique()))
            lut = dict(zip(x[i].unique(), col))
            colours += [x[i].map(lut)]
        colours = pd.concat(colours, axis = 1)
        return colours

    def _heatmap(
            self, 
            x,
            annotate_samples_by = None,
            annotate_features_by = None,
            **kwargs
        ):
        if annotate_samples_by is not None:
            assert self.data.columns.tolist() == annotate_samples_by.index.tolist(), \
                "annotate_samples_by index must be same as data columns"
        if annotate_features_by is not None:
            assert self.data.index.tolist() == annotate_features_by.index.tolist(), \
                "annotate_features_by index must be same as data columns"
        sns.clustermap(
            x, 
            col_colors = annotate_samples_by, 
            row_colors = annotate_features_by,
            **kwargs
        )

    def _density_plot(
            self, 
            samples = None, 
            **kwargs
        ):
        if samples is None:
            samples = self.colnames
        sns.kdeplot(data = self.data[samples], **kwargs)
        plt.xlabel("Feature value")
