from pydata.ldata import ldata
from pydata.drdata import drdata
import pandas as pd
from umap import UMAP
import re
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


class umap(drdata):
    """
    Class to perform and store results from uniform manifold approximation
    and projection (UMAP)
    """

    def __init__(self, data, description, annotation, scaling=None):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame of UMAP projections for ncol samples and nrow
            projections.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching column
            names of data attribute.
        annotation: pandas.DataFrame
            A DataFrame of UMAP projection annotation.
        scaling: str
            String describing the scaling procedure used before UMAP e.g.
            zscore.
        """
        super().__init__(data, description, annotation)
        self._scaling = scaling

    def __str__(self):
        out = super().__str__()
        out = re.sub("features", "UMAP projections", out)
        return out + f"\n - Scaling: {self.scaling}"

    def __repr__(self):
        out = super().__repr__()
        out = re.sub("features", "UMAP projections", out)
        return out + f"\n - Scaling: {self.scaling}"

    def _get_rownames(self):
        return super(umap, self)._get_rownames()

    def _set_rownames(self, value: list):
        """
        Set feature names for umap object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert all(
            [bool(re.search("^UMAP\\d+", i)) for i in value]
        ), "rownames must be in format UMAP1, UMAP2, etc"
        super(umap, self)._set_rownames(value)

    rownames = property(_get_rownames, _set_rownames)

    def _get_scaling(self):
        return getattr(self, "_scaling")

    def _set_scaling(self, value: str):
        self._scaling = value

    scaling = property(_get_scaling, _set_scaling)

    def _validate(self):
        assert all(
            [bool(re.search("^UMAP\d+", i)) for i in self.data.index]
        ), "rownames must be in format UMAP1, UMAP2, etc"
        super()._validate()

    @staticmethod
    def analyse(data: ldata, n_comp: int = 2, scaling: str = "Zscore", **kwargs):
        """
        Parameters
        ----------
        data: ldata object
        n_comp: Number of UMAP projections to compute. Default is 2.
        scaling: Scaling method before UMAP calculation. Default is "Zscore".
        **kwargs: Passed to UMAP method.
        """
        dat = deepcopy(data.data.transpose())
        match scaling:
            case "none":
                dat = dat
            case "Zscore":
                dat = StandardScaler().fit_transform(dat)
            case _:
                raise Exception(scaling + " scaling method not implemented")

        u = UMAP(n_components=n_comp, random_state=42, **kwargs)
        fit = u.fit_transform(dat)
        fit = pd.DataFrame(fit, columns=["UMAP" + str(i) for i in range(1, n_comp + 1)])
        fit.index = data.description["ID"].tolist()
        out = umap(
            data=fit.transpose(),
            description=data.description,
            annotation=pd.DataFrame(fit.columns.tolist(), columns=["ID"]),
            scaling=scaling,
        )
        return out
