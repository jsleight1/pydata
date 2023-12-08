import pandas as pd

class ldata:
    """
    Simple dataset computation and visualisation.

    ldata objects use a L-shaped data structure where the underlying 
    data set is a DataFrame of numeric values with an associated 
    description and annotation DataFrame containing relevant sample 
    and feature metadata respectively.

    Examples
    --------

    >>> # Generate some random data
    >>> np.random.seed(38)
    >>> data = pd.DataFrame(
    >>>     np.random.randint(0, 10, size = 100).reshape(20, 5),
    >>>     index = ["Feature" + str(i)for i in range(1, 21)], 
    >>>     columns = ["Sample" + str(i)for i in range(1, 6)] 
    >>> )
    >>>
    >>> # Generate sample metadata
    >>> desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
    >>>
    >>> # Generate feature metadata
    >>> annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})
    >>>
    >>> # Generate ldata object
    >>> x = ldata(data, desc, annot)
    >>> x
    >>> x.data
    >>> x.description
    >>> x.annotation
    """
    def __init__(self, data, description, annotation):
        """
        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pd.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        annotation : pd.DataFrame
            A DataFrame of feature annotation with ID column matching 
            row names of data attribute.
        """

        self._data = data 
        self._description = description
        self._annotation = annotation
        self._validate()

    def __str__(self):
        return (
            f"ldata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def __repr__(self):
        return (
            f"ldata object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"
        )

    def _get_data(self):
        return self._data
    def _set_data(self, value: pd.DataFrame):
        """
        Set data attribute for ldata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        """
        assert isinstance(value, pd.DataFrame), "data is not DataFrame"
        self._check_dimnames(data = value)
        self._data = value
    data = property(_get_data, _set_data)

    def _get_description(self):
        return self._description
    def _set_description(self, value: pd.DataFrame):
        """
        Set description attribute for ldata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of sample descriptions with ID column matching 
            columns names of data attribute.
        """
        self._check_dimnames(description = value)
        self._description = value
    description = property(_get_description, _set_description)

    def _get_annotation(self):
        return self._annotation
    def _set_annotation(self, value: pd.DataFrame):
        """
        Set description attribute for ldata object.
        ------------------------------------
        value: pandas.core.frame.DataFrame
            A DataFrame of feature annotation with ID column matching 
            row names of data attribute.
        """
        self._check_dimnames(annotation = value)
        self._annotation = value
    annotation = property(_get_annotation, _set_annotation)

    def _get_rownames(self):
        return self.data.index.values.tolist()
    def _set_rownames(self, value: list):
        """
        Set feature names for ldata object.
        ------------------------------------
        value: list
            A list of feature names.
        """
        assert isinstance(value, list), "value must be list"
        assert len(value) == self.data.shape[0], "value does not match data dims"
        assert len(value) == len(set(value)), "value must contain unique IDs"
        self.annotation["ID"] = value
        self.data.index = value
    rownames = property(_get_rownames, _set_rownames)

    def _get_colnames(self):
        return self.data.columns.tolist()
    def _set_colnames(self, value: list):
        """
        Set sample names for ldata object.
        ------------------------------------
        value: list
            A list of sample names.
        """
        assert isinstance(value, list), "value must be list"
        assert len(value) == self.data.shape[1], "value does not match data dims"
        assert len(value) == len(set(value)), "value must contain unique IDs"
        self.description["ID"] = value
        self.data.columns = value
    colnames = property(_get_colnames, _set_colnames)

    def _get_dimnames(self):
        return [self.rownames, self.colnames]
    def _set_dimnames(self, value: list):
        assert isinstance(value, list), "value must be list"
        assert len(value) == 2, "value must be list with rownames and colnames"
        self.rownames = value[0]
        self.colnames = value[1]
    dimnames = property(_get_dimnames, _set_dimnames)

    def _validate(self):
        assert isinstance(self.data, pd.DataFrame), "data is not DataFrame"
        assert isinstance(self.description, pd.DataFrame), \
            "description is not DataFrame"
        assert isinstance(self.annotation, pd.DataFrame), \
            "annotation is not DataFrame"
        self._check_dimnames()

    def _check_dimnames(
            self, 
            data: pd.DataFrame = None, 
            description: pd.DataFrame = None, 
            annotation: pd.DataFrame = None
        ):
        if data is None:
            data = self.data 
        if description is None:
            description = self.description
        if annotation is None:
            annotation = self.annotation
        rownames = data.index.values.tolist()
        colnames = data.columns.tolist()
        assert len(rownames) == len(set(rownames)), \
            "rownames must contain unique IDs"
        assert len(colnames) == len(set(colnames)), \
            "colnames must contain unique IDs"
        assert rownames == annotation["ID"].tolist(), \
            "data rownames do not match annotation ID"
        assert colnames == description["ID"].tolist(), \
            "data colnames do not match description ID"


    