from pydata.ldata import ldata
import re
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class drdata(ldata):
    """
    Class to store results from a dimension reduction analysis
    """

    def __init__(self, data, description, annotation, scaling=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame of reduce dimensions for ncol samples and nrow
            components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of reduce dimensions annotation
        scaling: str
            String describing the scaling procedure used before PCA e.g. "None", "zscore".
        """

        super().__init__(data, description, annotation)
        self._scaling = scaling
        self._validate()

    def __str__(self):
        out = super().__str__()
        out = re.sub("features", f"{super()._format_type()} components", out)
        out = out + f"\n - Scaling: {self.scaling}"
        return out

    def __repr__(self):
        out = super().__str__()
        out = re.sub("features", f"{super()._format_type()} components", out)
        out = out + f"\n - Scaling: {self.scaling}"
        return out

    def _get_rownames(self):
        return super(drdata, self)._get_rownames()

    def _set_rownames(self, value: list):
        """Set feature names for drdata object.

        Parameters
        ----------
        value: list
            A list of feature names.
        """
        t = super()._format_type().upper()
        assert all(
            [bool(re.search(f"^{t}\\d+", i)) for i in value]
        ), f"rownames must be in format {t}1, {t}2, etc"
        super(drdata, self)._set_rownames(value)

    rownames = property(_get_rownames, _set_rownames)

    def _get_scaling(self):
        return getattr(self, "_scaling")

    def _set_scaling(self, value: str):
        self._scaling = value

    scaling = property(_get_scaling, _set_scaling)

    @staticmethod
    def scale(data: ldata, method: str = "none", **kwargs):
        """Scale ldata object

        Parameters
        ----------
        data: ldata object.
        method: Scaling method. Options: "none", "zscore". Default is "none".
        **kwargs: Passed to methods.

        Returns
        ---------
        pd.DataFrame of scaled data.
        """
        dat = deepcopy(data.data.transpose())
        match method:
            case "none":
                dat = dat
            case "zscore":
                dat = pd.DataFrame(
                    StandardScaler().fit_transform(dat),
                    index=data.colnames,
                    columns=data.rownames,
                )
            case _:
                raise Exception(method + " scaling method not implemented")
        return dat

    def _validate(self):
        t = super()._format_type().upper()
        assert all(
            [bool(re.search(f"^{t}\\d+", i)) for i in self.rownames]
        ), f"rownames must be in format {t}1, {t}2, etc"
        super()._validate()

    def subset(self):
        raise Exception(f"Cannot subset {super()._format_type()} object")

    def transpose(self):
        raise Exception(f"Cannot transpose {super()._format_type()} object")

    def concat(self, objs=[]):
        raise Exception(f"Cannot concat {super()._format_type()} object")

    def plot(self, type: str = "scatter", **kwargs):
        """Plot drdata object.

        Parameters
        ----------
        type: str
            Type of plot. Default is "scatter".
        **kwargs:
            Passed to plotting methods.
        """
        self._validate()
        match type:
            case "scatter":
                self._scatter_plot(**kwargs)

    def _scatter_plot(
        self,
        xaxis=None,
        yaxis=None,
        colour_by: str = "ID",
        interactive: bool = False,
        **kwargs,
    ):
        t = super()._format_type().upper()
        if xaxis is None:
            xaxis = t + "1"
        if yaxis is None:
            yaxis = t + "2"
        df = deepcopy(self.data.transpose().reset_index(names="ID"))
        df = pd.merge(df, self.description, on="ID")
        if interactive:
            px.scatter(
                df, x=xaxis, y=yaxis, color=colour_by, hover_name="ID", **kwargs
            ).show()
        else:
            sns.relplot(data=df, x=xaxis, y=yaxis, hue=colour_by, **kwargs)
            plt.show()
