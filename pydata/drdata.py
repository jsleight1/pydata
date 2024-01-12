from pydata.ldata import ldata
import re


class drdata(ldata):
    """
    Class to store results from a dimension reduction analysis
    """

    def __init__(self, data, description, annotation):
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
        """

        super().__init__(data, description, annotation)

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
