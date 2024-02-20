import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import re


class ldata:
    """L-shared data structure object.

    ldata objects use a L-shaped data structure where the underlying
    data set is a DataFrame of numeric values with an associated
    description and annotation DataFrames containing relevant sample
    and feature metadata respectively.

    Examples
    ----------

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
        data : pandas.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation: pandas.DataFrame
            A DataFrame of feature annotation with ID column matching
            row names of data attribute.
        """

        self._data = deepcopy(data)
        self._description = deepcopy(description)
        self._annotation = deepcopy(annotation)
        self._validate()

    def __str__(self):
        t = self._format_type()
        return f"{t} object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"

    def __repr__(self):
        t = self._format_type()
        return f"{t} object:\n - Dimensions: {self.data.shape[1]} (samples) x {self.data.shape[0]} (features)"

    def _get_data(self):
        return self._data

    def _set_data(self, value: pd.DataFrame):
        """Set data attribute for ldata object.

        Parameters
        ----------
        value: pandas.core.frame.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        """
        assert isinstance(value, pd.DataFrame), "data is not DataFrame"
        self._check_dimnames(data=value)
        self._data = value

    data = property(_get_data, _set_data)

    def _get_description(self):
        return self._description

    def _set_description(self, value: pd.DataFrame):
        """Set description attribute for ldata object.

        Parameters
        ----------
        value: pandas.core.frame.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        """
        self._check_dimnames(description=value)
        self._description = value

    description = property(_get_description, _set_description)

    def _get_annotation(self):
        return self._annotation

    def _set_annotation(self, value: pd.DataFrame):
        """Set description attribute for ldata object.

        Parameters
        ----------
        value: pandas.core.frame.DataFrame
            A DataFrame of feature annotation with ID column matching
            row names of data attribute.
        """
        self._check_dimnames(annotation=value)
        self._annotation = value

    annotation = property(_get_annotation, _set_annotation)

    def _get_rownames(self):
        return self.data.index.values.tolist()

    def _set_rownames(self, value: list):
        """Set feature names for ldata object.

        Parameters
        ----------
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
        """Set sample names for ldata object.

        Parameters
        ----------
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
        """Set dimnames for ldata object

        Parameters
        ----------
        value: list
            A list of lists of rownames and colnames
        """
        assert isinstance(value, list), "value must be list"
        assert len(value) == 2, "value must be list with rownames and colnames"
        self.rownames = value[0]
        self.colnames = value[1]

    dimnames = property(_get_dimnames, _set_dimnames)

    def _validate(self):
        assert isinstance(self.data, pd.DataFrame), "data is not DataFrame"
        assert all(
            self.data.dtypes.map(pd.api.types.is_numeric_dtype)
        ), "data must all be numeric values"
        assert isinstance(
            self.description, pd.DataFrame
        ), "description is not DataFrame"
        assert isinstance(self.annotation, pd.DataFrame), "annotation is not DataFrame"
        self._check_dimnames()

    def _check_dimnames(
        self,
        data: pd.DataFrame = None,
        description: pd.DataFrame = None,
        annotation: pd.DataFrame = None,
    ):
        if data is None:
            data = self.data
        if description is None:
            description = self.description
        if annotation is None:
            annotation = self.annotation
        rownames = data.index.values.tolist()
        colnames = data.columns.tolist()
        assert len(rownames) == len(set(rownames)), "rownames must contain unique IDs"
        assert len(colnames) == len(set(colnames)), "colnames must contain unique IDs"
        assert (
            rownames == annotation["ID"].tolist()
        ), "data rownames do not match annotation ID"
        assert (
            colnames == description["ID"].tolist()
        ), "data colnames do not match description ID"

    @staticmethod
    def example_ldata(type: str = "iris", **kwargs):
        """Generate example ldata object

        Generate example ldata object. Options for example data sets include
        the "iris" dataset or one can also generate a "simulate" dataset.
        See ldata_.simulate_ldata for details.

        Parameters
        ----------
        type: str
            Type of example to generate. Either "iris" or "simulate" datasets.

        Returns
        ----------
        ldata object.

        Examples
        ----------
        >>> x = ldata.example_ldata()
        >>> print(x)
        """
        match type:
            case "iris":
                return ldata._iris_ldata(**kwargs)
            case "simulate":
                return ldata._simulate_ldata(**kwargs)
            case _:
                raise Exception(type + " type data not available")

    @staticmethod
    def _iris_ldata(**kwargs):
        iris = sns.load_dataset("iris")
        desc = pd.DataFrame(
            {
                "ID": ["Sample" + str(i) for i in range(1, 151)],
                "Species": iris["species"],
            }
        )
        data = iris.drop(["species"], axis=1).transpose()
        data.columns = desc["ID"].tolist()
        annot = pd.DataFrame({"ID": data.index.tolist()})
        annot["type"] = annot["ID"].str.extract(r"_(.*)$", expand=False)
        return ldata(data, desc, annot)

    @staticmethod
    def _simulate_ldata(
        min: float = 0,
        max: float = 10,
        nsamples: int = 5,
        nfeatures: int = 20,
        **kwargs,
    ):
        """Generate simulated example ldata object.

        This simulates data from a uniform distribution between minimum and
        maximum values for nfeatures and nsamples.

        Parameters
        ----------
        min: float
            Minimum value of uniform distribution
        max: float
            Maximum value to uniform distribution
        nsamples: int
            Number of samples in ldata object.
        nfeatures: int
            Number of features in ldata object.

        Returns
        ----------
        ldata object
        """
        np.random.seed(38)
        data = pd.DataFrame(
            np.random.uniform(min, max, (nfeatures, nsamples)),
            index=["Feature" + str(i) for i in range(1, nfeatures + 1)],
            columns=["Sample" + str(i) for i in range(1, nsamples + 1)],
        )
        desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, nsamples + 1)]})
        annot = pd.DataFrame(
            {"ID": ["Feature" + str(i) for i in range(1, nfeatures + 1)]}
        )
        return ldata(data, desc, annot)

    def subset(self, samples=None, features=None):
        """Subset ldata object

        Parameters
        ----------
        samples:
            Samples to subset ldata object to.
        features:
            Features to subset ldata object to.

        Returns
        ----------
        ldata object.

        Examples
        ----------
        >>> x = ldata.example_ldata()
        >>> x.subset(samples = ["Sample1"])
        """
        if samples is None:
            samples = self.colnames
        if features is None:
            features = self.rownames

        assert set(samples).issubset(self.colnames), "samples are not in data"
        assert set(features).issubset(self.rownames), "features are not in data"

        self = deepcopy(self)
        new_dat = deepcopy(self.data)
        new_dat = new_dat[samples][new_dat.index.isin(features)]
        new_desc = deepcopy(self.description)
        new_desc = new_desc[new_desc["ID"].isin(samples)].reset_index(drop=True)
        new_annot = deepcopy(self.annotation)
        new_annot = new_annot[new_annot["ID"].isin(features)].reset_index(drop=True)

        self._data = new_dat
        self._description = new_desc
        self._annotation = new_annot
        self._validate()
        return self

    def transpose(self):
        """Transpose ldata object

        Similar to pandas.DataFrame.transpose where columns of ldata object
        become rows and rows become columns but includes correct ldata
        description and annotation switching.

        Returns
        ----------
        ldata object.

        Examples
        ----------
        >>> x = ldata.example_ldata()
        >>> x.transpose()
        """
        self = deepcopy(self)
        new_dat = deepcopy(self.data)
        new_desc = deepcopy(self.description)
        new_annot = deepcopy(self.annotation)
        self._data = new_dat.transpose()
        self._description = new_annot
        self._annotation = new_desc
        self._validate()
        return self

    def concat(self, *objs):
        """Concatenate samples from multiple ldata objects

        Similar to pandas.DataFrame.concat where ldata objects
        are bound to each other to allow the addition of samples only.
        The feature set must be shared between ldata objects.

        Parameters
        ----------
        *objs:
            ldata objects.

        Returns
        ----------
        ldata object.

        Examples
        ----------
        >>> a = ldata.example_ldata(type="simulate")
        >>> b = ldata.example_ldata(type="simulate", min=5, max=6)
        >>> c = ldata.example_ldata(type="simulate", min=15, max=20)
        >>> b.colnames = ["Sample" + str(i) for i in range(6, 11)]
        >>> c.colnames = ["Sample" + str(i) for i in range(11, 16)]
        >>> a.concat(b, c)
        """
        assert all(
            [isinstance(i, type(self)) for i in objs]
        ), "objects must all be of same class"
        assert all(
            [i.rownames == self.rownames for i in objs]
        ), "objects must have same feature IDs"
        self = deepcopy(self)
        self._data = pd.concat([self.data] + [i.data for i in objs], axis=1)
        self._description = pd.concat(
            [self.description] + [i.description for i in objs]
        ).reset_index(drop=True)
        self._validate()
        return self

    def _format_type(self):
        return re.findall("'([^']*)'", str(type(self)))[0].split(".")[-1]
