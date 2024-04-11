import pytest
from pydata.pydata import pydata
from pydata.pca import pca
from pydata.lda import lda
from pydata.tsne import tsne
import numpy as np
import pandas as pd

np.random.seed(38)
data = pd.DataFrame(
    np.random.randint(0, 10, size=120).reshape(20, 6),
    index=["Feature" + str(i) for i in range(1, 21)],
    columns=["Sample" + str(i) for i in range(1, 7)],
)
grps = np.array(["Control", "Treatment"])
desc = pd.DataFrame(
    {
        "ID": ["Sample" + str(i) for i in range(1, 7)],
        "Treatment": np.repeat(grps, [3, 3], axis=0),
    }
)
annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})


def test_pydata_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        pydata(data.head(2), desc, annot)
    assert "data rownames do not match annotation ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        pydata(data, desc.head(2), annot)
    assert "data colnames do not match description ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        pydata(data, desc, annot.head(2))
    assert "data rownames do not match annotation ID" in str(err.value)

    x = pydata(data, desc, annot)

    assert isinstance(x, pydata)
    assert x.data.equals(data)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)

    snapshot.assert_match(str(x), "pydata.txt")


def test_example_pydata(snapshot):
    x = pydata.example_pydata()
    assert isinstance(x, pydata)
    snapshot.assert_match(str(x), "example_pydata.txt")


def test_perform_dimension_reduction(snapshot):
    x = pydata(data, desc, annot)

    with pytest.raises(Exception) as err:
        x.perform_dimension_reduction("dr")
    assert "dr dimension reduction not implemented" in str(err.value)

    x.perform_dimension_reduction("pca")
    x.perform_dimension_reduction("lda", target="Treatment", n_comp=1)

    assert isinstance(x.pcs, pca)
    assert isinstance(x.lda, lda)


def test_subset():
    x = pydata(data, desc, annot)

    with pytest.raises(AssertionError) as err:
        x.subset(samples=["sample"])
    assert "samples are not in data" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.subset(features=["feature"])
    assert "features are not in data" in str(err.value)

    x.perform_dimension_reduction("pca")
    x.perform_dimension_reduction("lda", target="Treatment", n_comp=1)
    samples = ["Sample1", "Sample3"]
    features = ["Feature1", "Feature3", "Feature18"]
    s = x.subset(samples=samples, features=features)

    assert isinstance(s, pydata)
    assert s.colnames == samples
    assert s.rownames == features
    assert s.description["ID"].tolist() == samples
    assert s.description.index.tolist() == [0, 1]
    assert s.annotation["ID"].tolist() == features
    assert s.annotation.index.tolist() == [0, 1, 2]
    assert s.data.columns.tolist() == samples
    assert s.data.index.tolist() == features
    assert s.pcs is None
    assert s.lda is None
    assert x.colnames == data.columns.tolist()
    assert x.rownames == data.index.tolist()


def test_transpose():
    x = pydata(data, desc, annot)
    x.perform_dimension_reduction("pca")
    x.perform_dimension_reduction("lda", target="Treatment", n_comp=1)

    l = x.transpose()

    assert isinstance(l, pydata)
    assert l.colnames == x.rownames
    assert l.rownames == x.colnames
    assert l.description["ID"].tolist() == x.annotation["ID"].tolist()
    assert l.annotation["ID"].tolist() == x.description["ID"].tolist()
    assert l.data.columns.tolist() == x.data.index.tolist()
    assert l.data.index.tolist() == x.data.columns.tolist()
    assert l.pcs is None
    assert l.lda is None
    assert x.colnames == data.columns.tolist()
    assert x.rownames == data.index.tolist()
