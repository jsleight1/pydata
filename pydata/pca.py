from pydata.drdata import drdata
import pandas as pd
import re


class pca(drdata):
    """
    Class to store results from principal component analysis (PCA)
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
            columns names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of PC annotation with ID column matching
            row names of data attribute e.g. PC1, PC2, etc
        scaling: str
            String describing the scaling procedure used before PCA e.g. Zscore
        method: str
            String describing the method used to perform PCA e.g. SVD
        """

        super().__init__(data, description, annotation)
        self._scaling = scaling
        self._method = method
        self._validate()

    def __str__(self):
        out = super().__str__()
        out = re.sub("features", "principal components", out)
        return out + f"\n - Scaling: {self.scaling}\n - Method: {self.method}"

    def __repr__(self):
        out = super().__repr__()
        out = re.sub("features", "principal components", out)
        return out + f"\n - Scaling: {self.scaling}\n - Method: {self.method}"

    def _get_method(self):
        return getattr(self, "_method")

    def _set_method(self, value: str):
        self._method = value

    method = property(_get_method, _set_method)

    def _get_scaling(self):
        return getattr(self, "_scaling")

    def _set_scaling(self, value: str):
        self._scaling = value

    scaling = property(_get_scaling, _set_scaling)

    def _get_annotation(self):
        return super(pca, self)._get_annotation()

    def _set_annotation(self, value: pd.DataFrame):
        """
        Set description attribute for ldata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of PC annotation with ID column matching
            row names of data attribute.
        """
        assert all(
            [bool(re.search("^PC\\d+", i)) for i in value["ID"]]
        ), "ID column must be in format PC1, PC2, etc"
        assert (
            "Percentage variance explained" in value.columns
        ), "annotation must contain 'Percentage variance explained' column"
        super(pca, self)._set_annotation(value)

    annotation = property(_get_annotation, _set_annotation)

    def _get_rownames(self):
        return super(pca, self)._get_rownames()

    def _set_rownames(self, value: list):
        """
        Set feature names for ldata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert all(
            [bool(re.search("^PC\\d+", i)) for i in value]
        ), "rownames must be in format PC1, PC2, etc"
        super(pca, self)._set_rownames(value)

    rownames = property(_get_rownames, _set_rownames)

    def _validate(self):
        assert all(
            [bool(re.search("^PC\\d+", i)) for i in self.data.index]
        ), "rownames must be in format PC1, PC2, etc"
        assert (
            "Percentage variance explained" in self.annotation.columns
        ), "annotation must contain 'Percentage variance explained' column"
        super()._validate()
