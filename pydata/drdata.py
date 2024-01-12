from pydata.ldata import ldata
import re
from copy import deepcopy
import pandas as pd 
import seaborn as sns


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
            String describing the scaling procedure used before PCA e.g. "None", "Zscore".
        """

        out = super().__init__(data, description, annotation)
        self._scaling = scaling
        self._validate()

    def __str__(self):
        out = super().__str__()
        out = re.sub("features", f"{super().format_type()} components", out)
        out = out + f"\n - Scaling: {self.scaling}"
        return out

    def __repr__(self):
        out = super().__str__()
        out = re.sub("features", f"{super().format_type()} components", out)
        out = out + f"\n - Scaling: {self.scaling}"
        return out

    def _get_rownames(self):
        return super(drdata, self)._get_rownames()

    def _set_rownames(self, value: list):
        """
        Set feature names for drdata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        t = super().format_type().upper()
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

    def _validate(self):
        t = super().format_type().upper()
        assert all(
            [bool(re.search(f"^{t}\\d+", i)) for i in self.rownames]
        ), f"rownames must be in format {t}1, {t}2, etc"
        super()._validate()

    def subset(self):
        raise Exception(f"Cannot subset {super().format_type()} object")

    def transpose(self):
        raise Exception(f"Cannot transpose {super().format_type()} object")

    def concat(self, objs=[]):
        raise Exception(f"Cannot concat {super().format_type()} object")

    def plot(self, xaxis=None, yaxis=None, colour_by: str = "ID", **kwargs):
        self._validate()
        t = super().format_type().upper()
        if xaxis is None:
            xaxis = t + "1"
        if yaxis is None:
            yaxis = t + "2"
        df = deepcopy(self.data.transpose().reset_index(names="ID"))
        df = pd.merge(df, self.description, on="ID")
        sns.relplot(data=df, x=xaxis, y=yaxis, hue=colour_by)
