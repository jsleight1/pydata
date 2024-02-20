from pydata.pydata import pydata
import pandas as pd
from rnanorm.datasets import load_toy_data, load_gtex
from rnanorm import CPM, TPM, FPKM, UQ, CUF, TMM, CTF
from pydeseq2.preprocessing import deseq2_norm
import os
from copy import deepcopy


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

    def __init__(
        self,
        data,
        description,
        annotation,
        gtf=None,
        filtering_method=None,
        normalisation_method=None,
    ):
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
        self._filtering_method = filtering_method
        self._normalisation_method = normalisation_method
        self._validate()

    def __str__(self):
        out = super().__str__()
        return (
            out
            + f"\n - gtf: {self.gtf}\n - Filter: {self.filtering_method}\n - normalisation: {self.normalisation_method}"
        )

    def __repr__(self):
        out = super().__repr__()
        return (
            out
            + f"\n - gtf: {self.gtf}\n - Filter: {self.filtering_method}\n - normalisation: {self.normalisation_method}"
        )

    def _get_gtf(self):
        return getattr(self, "_gtf")

    def _set_gtf(self, value: str):
        self._gtf = value

    gtf = property(_get_gtf, _set_gtf)

    def _get_normalisation_method(self):
        return getattr(self, "_normalisation_method")

    def _set_normalisation_method(self, value: str):
        self._normalisation_method = value

    normalisation_method = property(
        _get_normalisation_method, _set_normalisation_method
    )

    def _get_filtering_method(self):
        return getattr(self, "_filtering_method")

    def _set_filtering_method(self, value: str):
        self._filtering_method = value

    filtering_method = property(_get_filtering_method, _set_filtering_method)

    @staticmethod
    def example_rnadata(type: str = "toy", **kwargs):
        """Generate example rnadata.

        See rnanorm.datasets.load_toy_data() for details.

        Parameters
        ----------
        **kwargs:
            Passed to ldata.example_ldata

        Returns
        ----------
        rnadata object

        Examples
        ----------
        >>> x = rnadata.example_rnadata()
        >>> print(x)
        """
        match type:
            case "toy":
                dat = load_toy_data()
            case "gtex":
                dat = load_gtex()
            case _:
                raise Exception(type + "example rnadata not implemented")

        return rnadata(
            dat.exp.transpose(),
            pd.DataFrame({"ID": dat.exp.index}),
            pd.DataFrame({"ID": dat.exp.columns}),
            str(dat.gtf_path),
        )

    def filter_counts(self, method: str = "sum", thresh: int = 10, **kwargs):
        self._validate()
        out = deepcopy(self)
        match method:
            case "sum":
                keep = out.data.sum(axis=1) >= thresh
            case "mean":
                keep = out.data.mean(axis=1) >= thresh
            case "min":
                keep = out.data.min(axis=1) >= thresh
            case _:
                raise Exception(method + " filtering not implemented")
        out = out.subset(features=keep[keep].index)
        print(f"Dropping {self.data.shape[0] - len(keep)} features")
        out.filtering_method = method
        return out

    def normalise(self, method: str = "TMM", **kwargs):
        """Normalise rnadata object

        Implement RNAseq count based normalisation using associated functions
        from rnanorm package (https://pypi.org/project/rnanorm/) and pydeseq2
        (https://pypi.org/project/pydeseq2/).

        Parameters
        ----------
        method: str
            Type of normalisation to perform. Current implementations include
            counts per million normalisation (CPM), fragments per kilo-base
            million (FKPM) normalisation, transcripts per kilo-base million
            (TPM) normalisation, upper quartile (UQ) normalisation, counts
            adjusted with upper quartile factors normalisation, trimmed mean of
            M-values (TMM) normalisation, counts adjusted with TMM factors
            normalisation (CTF) and pyDESeq2's median of ratios normalisation.
            Default is TMM.
        **kwargs:
            Passed to relevant method in rnanorm.

        Returns
        ----------
        rnadata object

        Examples
        ----------
        >>> x = rnadata.example_rnadata()
        >>> norm_x = x.normalise(method="CPM")
        """
        self._validate()
        out = deepcopy(self)
        in_data = out.data.transpose()
        match method:
            case "TMM":
                out.data = (
                    TMM(**kwargs).set_output(transform="pandas").fit_transform(in_data)
                ).transpose()
            case "CPM":
                out.data = (
                    CPM(**kwargs).set_output(transform="pandas").fit_transform(in_data)
                ).transpose()
            case "TPM":
                assert os.path.isfile(out.gtf), "Does GFT file exist?"
                out.data = (
                    TPM(out.gtf, **kwargs)
                    .set_output(transform="pandas")
                    .fit_transform(in_data)
                ).transpose()
            case "FPKM":
                assert os.path.isfile(out.gtf), "Does GFT file exist?"
                out.data = (
                    FPKM(out.gtf, **kwargs)
                    .set_output(transform="pandas")
                    .fit_transform(in_data)
                ).transpose()
            case "UQ":
                out.data = (
                    UQ(**kwargs).set_output(transform="pandas").fit_transform(in_data)
                ).transpose()
            case "CUF":
                out.data = (
                    CUF(**kwargs).set_output(transform="pandas").fit_transform(in_data)
                ).transpose()
            case "CTF":
                out.data = (
                    CTF(**kwargs).set_output(transform="pandas").fit_transform(in_data)
                ).transpose()
            case "DESeq2":
                out.data = deseq2_norm(in_data)[0].transpose()
            case _:
                raise Exception(method + " normalisation not implemented")
        out.normalisation_method = method
        return out
