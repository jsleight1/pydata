from pydata.ldata import ldata
from pydata.drdata import drdata
import re
import pandas as pd
from sklearn.manifold import TSNE


class tsne(drdata):
    """
    Perform store results from t-distributed stochastic neighbor embedding
    (t-SNE)
    """

    def __init__(self, data, description, annotation, scaling=None):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame of t-SNE components data for ncol samples and nrow
            t-SNE components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            column names of data attribute.
        annotation: pandas.DataFrame
            A DataFrame of t-SNE components annotation.
        """
        super().__init__(data, description, annotation, scaling)

    @staticmethod
    def analyse(data, n_comp: int = 2, scaling: str = "zscore", **kwargs):
        """Perform t-SNE dimension reduction

        Parameters
        ----------
        data:
            pydata object.
        n_comp: int
            Number of t-SNE components to compute. Default is 2.
        scaling: str
            Scaling method before TSNE calculation. Default is "zscore".
        **kwargs:
            Passed to sklearn.manifold.TSNE

        Returns
        ----------
        tsne object

        Examples
        ----------
        >>> x = pydata.example_pydata()
        >>> tnse.analyse(x)
        """
        dat = drdata.scale(data=data, method=scaling)
        t = TSNE(n_components=n_comp, **kwargs)
        fit = t.fit_transform(dat)
        fit = pd.DataFrame(fit, columns=["TSNE" + str(i) for i in range(1, n_comp + 1)])
        fit.index = data.description["ID"].tolist()
        out = tsne(
            data=fit.transpose(),
            description=data.description,
            annotation=pd.DataFrame(fit.columns.tolist(), columns=["ID"]),
            scaling=scaling,
        )
        return out
