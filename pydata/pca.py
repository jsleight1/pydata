from pydata.ldata import ldata
from pydata.drdata import drdata
import pandas as pd
import numpy as np
import re
from copy import deepcopy
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
import plotly.express as px


class pca(drdata):
    """
    Class to perform and store results from principal component analysis (PCA)
    """

    def __init__(self, data, description, annotation, scaling=None, method=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame of principal component data for ncol samples and nrow
            principal components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            column names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of PC annotation with ID column matching
            row names of data attribute e.g. PC1, PC2, etc
        scaling: str
            String describing the scaling procedure used before PCA e.g. zscore
        method: str
            String describing the method used to perform PCA e.g. SVD
        """

        super().__init__(data, description, annotation, scaling)
        self._method = method
        self._validate()

    def __str__(self):
        out = super().__str__()
        return out + f"\n - Method: {self.method}"

    def __repr__(self):
        out = super().__repr__()
        return out + f"\n - Method: {self.method}"

    def _get_method(self):
        return getattr(self, "_method")

    def _set_method(self, value: str):
        self._method = value

    method = property(_get_method, _set_method)

    def _get_annotation(self):
        return super(pca, self)._get_annotation()

    def _set_annotation(self, value: pd.DataFrame):
        """
        Set description attribute for pca object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of PC annotation with ID column matching
            row names of data attribute.
        """
        assert all(
            [bool(re.search("^PCA\\d+", i)) for i in value["ID"]]
        ), "ID column must be in format PCA1, PCA2, etc"
        assert (
            "Percentage variance explained" in value.columns
        ), "annotation must contain 'Percentage variance explained' column"
        super(pca, self)._set_annotation(value)

    annotation = property(_get_annotation, _set_annotation)

    def _validate(self):
        assert (
            "Percentage variance explained" in self.annotation.columns
        ), "annotation must contain 'Percentage variance explained' column"
        super()._validate()

    def plot(self, type: str = "scatter", **kwargs):
        """
        Plot pca object.
        ---------------
        type: str
            Type of plot. Either "scatter" or "elbow", Default is "scatter".
        **kwargs:
            Passed to plotting methods.
        """
        self._validate()
        if type == "elbow":
            self._elbow_plot(**kwargs)
        else:
            super().plot(type=type, **kwargs)

    def _elbow_plot(self, n_comp=None, interactive: bool = False, **kwargs):
        if n_comp is None:
            n_comp = self.data.shape[0]
        if interactive:
            plot = px.line(
                self.annotation,
                x="ID",
                y="Percentage variance explained",
                markers=True,
                **kwargs,
            )
            plot.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None)
            plot.show()
        else:
            plot = sns.lineplot(
                data=self.annotation,
                x="ID",
                y="Percentage variance explained",
                marker="o",
                **kwargs,
            )
            plot.set(xlabel=None)

    @staticmethod
    def analyse(
        data,
        n_comp: int = 2,
        scaling: str = "zscore",
        method: str = "SVD",
        **kwargs,
    ):
        """
        Perform PCA dimension reduction
        -------------------------------
        data:
            pydata object.
        n_comp: int
            Number of principal component to compute. Default is 2.
        scaling: str
            Scaling method before PCA calculation. Default is "zscore".
        method: str
            PCA method for PCA calculation: Default is "SVD" for singular
            value decomposition.
        **kwargs:
            Passed to PCA method.
        """
        dat = drdata.scale(data=data, method=scaling)
        match method:
            case "SVD":
                pcs = pca._svd_pca(
                    x=dat, desc=data.description, n_comp=n_comp, **kwargs
                )
            case "Kernel":
                pcs = pca._kernel_pca(
                    x=dat, desc=data.description, n_comp=n_comp, **kwargs
                )
            case _:
                raise Exception(method + " pca method not implemented")

        pcs.scaling = scaling
        pcs.method = method
        return pcs

    @staticmethod
    def _svd_pca(x, desc, n_comp, **kwargs):
        p = PCA(n_components=n_comp, **kwargs)
        p_c = p.fit_transform(x)
        p_df = pd.DataFrame(
            data=p_c,
            columns=["PCA" + str(i) for i in range(1, n_comp + 1)],
            index=desc["ID"].tolist(),
        )
        var_expl = pd.DataFrame(
            {
                "ID": p_df.columns.tolist(),
                "Percentage variance explained": p.explained_variance_ratio_ * 100,
            }
        )
        out = pca(
            data=p_df.transpose(),
            description=desc,
            annotation=var_expl,
        )
        return out

    @staticmethod
    def _kernel_pca(x, desc, n_comp, **kwargs):
        p = KernelPCA(n_components=n_comp, **kwargs)
        p_c = p.fit_transform(x)
        p_df = pd.DataFrame(
            data=p_c,
            columns=["PCA" + str(i) for i in range(1, n_comp + 1)],
            index=desc["ID"].tolist(),
        )
        var_expl = pd.DataFrame(
            {
                "ID": p_df.columns.tolist(),
                "Percentage variance explained": (
                    (np.var(p_c, axis=0) / np.sum(np.var(p_c, axis=0))) * 100
                ),
            }
        )
        out = pca(
            data=p_df.transpose(),
            description=desc,
            annotation=var_expl,
        )
        return out
