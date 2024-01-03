from pydata.ldata import ldata
from pydata.drdata import drdata
import re
import pandas as pd
from sklearn.manifold import TSNE
from copy import deepcopy


class tsne(drdata):
    """
    Class to store results from t-distributed stochastic neighbor embedding
    (t-SNE)
    """

    def __init__(self, data, description, annotation):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame of t-SNE components data for ncol samples and nrow
            t-SNE components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of t-SNE components annotation
        """
        super().__init__(data, description, annotation)

    def _get_rownames(self):
        return super(tsne, self)._get_rownames()

    def _set_rownames(self, value: list):
        """
        Set feature names for ldata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert all(
            [bool(re.search("^TSNE\\d+", i)) for i in value]
        ), "rownames must be in format TSNE1, TSNE2, etc"
        super(tsne, self)._set_rownames(value)

    def _validate(self):
        assert all(
            [bool(re.search("^TSNE\\d+", i)) for i in self.data.index]
        ), "rownames must be in format TSNE1, TSNE2, etc"
        super()._validate()

    rownames = property(_get_rownames, _set_rownames)

    @staticmethod
    def analyse(data: ldata, n_comp: int = 2, **kwargs):
        """
        Parameters
        ----------
        n_comp: Number of t-SNE components to compute. Default is 2.
        **kwargs: Passed to sklearn.manifold.TSNE
        """

        t = TSNE(n_components=n_comp, **kwargs)
        fit = t.fit_transform(data.data.transpose())
        fit = pd.DataFrame(fit, columns=["TSNE" + str(i) for i in range(1, n_comp + 1)])
        fit.index = data.description["ID"].tolist()
        out = tsne(
            data=fit.transpose(),
            description=data.description,
            annotation=pd.DataFrame(fit.columns.tolist(), columns=["ID"]),
        )
        return out