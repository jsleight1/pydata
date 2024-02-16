from pydata.pydata import pydata
import pandas as pd
from rnanorm.datasets import load_toy_data
import os

class rnadata(pydata):
    """Bulk RNAseq count data computation and visualisation

    rnadata objects use a L-shaped data structure where the underlying
    data set is a DataFrame of numeric values with an associated
    description and annotation DataFrame containing relevant sample
    and feature metadata respectively.

    Examples
    --------

    >>> # Generate some random data
    >>> dat = load_toy_data()
    >>>
    >>> # Generate sample metadata
    >>> desc = pd.DataFrame({"ID": dat.exp.index})
    >>>
    >>> # Generate feature metadata
    >>> annot = pd.DataFrame({"ID": dat.exp.columns})
    >>>
    >>> # Generate pydata object
    >>> x = rnadata(dat.exp.transpose(), desc, annot, dat.gtf_path)
    >>> x
    >>> x.data
    >>> x.description
    >>> x.annotation
    >>> x.perform_dimension_reduction("pca")
    >>> x.plot("pca")
    >>> x.plot("distribution")
    """

    def __init__(self, data, description, annotation, gtf=None):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation: pandas.DataFrame
            A DataFrame of feature annotation with ID column matching
            row names of data attribute.
        gtf: str
            Optional argument describing path to gtf file used to 
            generate bulk RNAseq count data.
        """

        super().__init__(data, description, annotation)
        self._gtf = gtf
        self._validate()

    def __str__(self):
        out = super().__str__()
        return out + f"\n - gtf file: {self.gtf}"

    def __repr__(self):
        out = super().__repr__()
        return out + f"\n - gtf file: {self.gtf}"

    def _get_gtf(self):
        return getattr(self, "_gtf")

    def _set_gtf(self, value: str):
        self._gtf = value

    gtf = property(_get_gtf, _set_gtf)

    @staticmethod
    def example_rnadata(**kwargs):
        """Generate example rnadata.

        See rnanorm.datasets.load_toy_data() for details.

        Parameters
        ----------
        **kwargs:
            Passed to ldata.example_ldata

        Returns
        ----------
        pydata object

        Examples
        ----------
        >>> x = rnadata.example_rnadata()
        >>> print(x)
        """
        dat = load_toy_data()
        
        return rnadata(
            dat.exp.transpose(), 
            pd.DataFrame({"ID": dat.exp.index}), 
            pd.DataFrame({"ID": dat.exp.columns}), 
            str(dat.gtf_path)
        )

