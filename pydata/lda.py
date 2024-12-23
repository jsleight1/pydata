from pydata.ldata import ldata
from pydata.drdata import drdata
import re
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from copy import deepcopy


class lda(drdata):
    """
    Perform and store results from linear discriminant analysis (LDA)
    """

    def __init__(self, data, description, annotation, scaling=None, target=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame of LDA components data for ncol samples and nrow
            LDA components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            column names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of LDA components annotation
        target: str
            String describing target variable for LDA calculations
        """

        super().__init__(data, description, annotation, scaling)
        self._target = target

    def __str__(self):
        out = super().__str__()
        return out + f"\n - Target: {self.target}"

    def __repr__(self):
        out = super().__repr__()
        return out + f"\n - Target: {self.target}"

    def _get_target(self):
        return getattr(self, "_target")

    def _set_target(self, value: str):
        self._target = value

    target = property(_get_target, _set_target)

    def plot(self, colour_by=None, **kwargs):
        if colour_by is None:
            colour_by = self.target
        super().plot(type="scatter", colour_by=colour_by, **kwargs)

    @staticmethod
    def analyse(data, target: str, n_comp: int = 2, scaling: str = "zscore", **kwargs):
        """Perform LDA dimension reduction

        Parameters
        ----------
        data:
            pydata object.
        target: str
            String indicating the classifier variable to use for LDA.
        n_comp: int
            Number of LDA components to compute. Default is 2.
        scaling: str
            Scaling method before LDA calculation. Default is "zscore".
        **kwargs:
            Passed to sklearn.discriminant_analysis.LinearDiscriminantAnalysis.

        Returns
        ----------
        lda object

        Examples
        ----------
        >>> x = pydata.example_pydata()
        >>> lda.analyse(x, target = "Species")
        """
        assert target in data.description.columns, target + " is not in description"
        target_df = deepcopy(data.description[target])
        dat = drdata.scale(data=data, method=scaling)
        l = LinearDiscriminantAnalysis(n_components=n_comp, **kwargs)
        fit = l.fit(dat, target_df).transform(dat)
        fit = pd.DataFrame(fit, columns=["LDA" + str(i) for i in range(1, n_comp + 1)])
        fit.index = data.description["ID"].tolist()
        out = lda(
            data=fit.transpose(),
            description=data.description,
            annotation=pd.DataFrame(fit.columns.tolist(), columns=["ID"]),
            target=target,
            scaling=scaling,
        )
        return out
