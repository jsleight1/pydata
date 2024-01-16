from pydata.ldata import ldata
from pydata.drdata import drdata
import pandas as pd
from umap import UMAP
import re
from copy import deepcopy


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
        super().__init__(data, description, annotation, scaling)

    @staticmethod
    def analyse(data, n_comp: int = 2, scaling: str = "zscore", **kwargs):
        """
        Perform UMAP dimension reduction
        --------------------------------
        data:
            pydata object.
        n_comp: int
            Number of UMAP projections to compute. Default is 2.
        scaling: str
            Scaling method before UMAP calculation. Default is "zscore".
        **kwargs:
            Passed to UMAP method.
        """
        dat = drdata.scale(data=data, method=scaling)
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
