import pytest
from pydata.umap import umap
from pydata.pydata import pydata
from copy import deepcopy
import numpy as np
import pandas as pd

np.random.seed(38)
up = pd.DataFrame(
    np.random.randn(5, 5),
    index=["UMAP" + str(i) for i in range(1, 6)],
    columns=["Sample" + str(i) for i in range(1, 6)],
)
desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
annot = pd.DataFrame({"ID": ["UMAP" + str(i) for i in range(1, 6)]})


def test_umap_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        umap(up.head(2), desc, annot)
    assert "data rownames do not match annotation ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        umap(up, desc.head(2), annot)
    assert "data colnames do not match description ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        umap(up, desc, annot.head(2))
    assert "data rownames do not match annotation ID" in str(err.value)

    x = umap(up, desc, annot)

    assert isinstance(x, umap)
    assert x.data.equals(up)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)

    snapshot.assert_match(str(x), "umap.txt")


def test_annotation(snapshot):
    x = umap(up, desc, annot)
    new_annot = deepcopy(annot)
    new_annot["ID"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(AssertionError) as err:
        x.annotation = annot.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    new_annot = deepcopy(annot)
    new_annot["col"] = ["a", "b", "c", "d", "e"]
    x.annotation = new_annot
    assert x.annotation.equals(new_annot)
    snapshot.assert_match(x.annotation.to_csv(), "annotation.txt")


def test_dimnames():
    x = umap(up, desc, annot)

    assert x.dimnames == [x.rownames, x.colnames]

    with pytest.raises(AssertionError) as err:
        x.colnames = "A"
    assert "value must be list" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.colnames = ["A", "B"]
    assert "value does not match data dims" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.colnames = ["A", "B", "C", "C", "E"]
    assert "value must contain unique IDs" in str(err.value)

    x.colnames = ["A", "B", "C", "D", "E"]
    assert x.colnames == ["A", "B", "C", "D", "E"]
    assert x.description["ID"].tolist() == ["A", "B", "C", "D", "E"]

    with pytest.raises(AssertionError) as err:
        x.dimnames = "A"
    assert "value must be list" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.dimnames = ["A", "B", "C", "D", "E"]
    assert "value must be list with rownames and colnames" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.dimnames = [["A", "B", "C", "D", "E"], x.colnames]
    assert "rownames must be in format UMAP1, UMAP2, etc" in str(err.value)

    x.dimnames = [x.rownames, ["A", "B", "C", "D", "E"][::-1]]
    assert x.colnames == ["A", "B", "C", "D", "E"][::-1]
    assert x.description["ID"].tolist() == ["A", "B", "C", "D", "E"][::-1]


def test_subset():
    x = umap(up, desc, annot)
    with pytest.raises(Exception) as err:
        x.subset()
    assert "Cannot subset umap object" in str(err.value)


def test_transpose():
    x = umap(up, desc, annot)
    with pytest.raises(Exception) as err:
        x.transpose()
    assert "Cannot transpose umap object" in str(err.value)


def test_concat():
    a = umap(up, desc, annot)
    b = umap(up, desc, annot)
    with pytest.raises(Exception) as err:
        a.concat([b])
    assert "Cannot concat umap object" in str(err.value)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyse(snapshot):
    x = pydata.example_pydata().subset(["Sample1", "Sample2", "Sample3", "Sample4"])

    out = umap.analyse(x, n_neighbors=2)

    assert isinstance(out, umap)
    snapshot.assert_match(str(out), "umap_print.txt")
    snapshot.assert_match(out.data.round(3).to_csv(), "umap_data.txt")
