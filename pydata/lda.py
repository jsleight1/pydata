from pydata.ldata import ldata
import pandas as pd
import re

class lda(ldata): 
    """
    Class to store results from linear discriminant analysis (LDA)
    """

    def __init__(
            self, 
            data, 
            description, 
            annotation, 
            target = None
        ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame of LDA component data for ncol samples and nrow 
            LDA components.
        description: pd.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        annotation : pd.DataFrame
            A DataFrame of LDA components annotation
        target: str 
            String describing target variable for LDA calculations
        """

        super().__init__(data, description, annotation)
        self._target = target

    def __str__(self):
        return (
            f"lda object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (LDA components)\n - Target: {self.target}"
        )

    def __repr__(self):
        return (
            f"lda object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (LDA components)\n - Target: {self.target}"
        )
    
    def _get_target(self):
        return getattr(self, "_target")
    def _set_target(self, value: str):
        self._target = value
    target = property(_get_target, _set_target)
    
    def _get_rownames(self):
        return super(lda, self)._get_rownames()
    def _set_rownames(self, value: list):
        """
        Set feature names for ldata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert all([bool(re.search("^LD\\d+", i)) for i in value]), \
            "rownames must be in format LD1, LD2, etc"
        super(lda, self)._set_rownames(value)
    rownames = property(_get_rownames, _set_rownames)

    def _validate(self):
        assert all([bool(re.search("^LD\\d+", i)) for i in self.data.index]), \
            "rownames must be in format LD1, LD2, etc"
        super()._validate()

    

    