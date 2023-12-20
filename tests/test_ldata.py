import pytest
from pydata.ldata import ldata
from copy import deepcopy
import numpy as np
import pandas as pd

np.random.seed(38)
data = pd.DataFrame(
    np.random.randint(0, 10, size=100).reshape(20, 5),
    index=["Feature" + str(i) for i in range(1, 21)],
    columns=["Sample" + str(i) for i in range(1, 6)],
)
desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})


def test_ldata_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        ldata(data.head(2), desc, annot)
    assert "data rownames do not match annotation ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        ldata(data, desc.head(2), annot)
    assert "data colnames do not match description ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        ldata(data, desc, annot.head(2))
    assert "data rownames do not match annotation ID" in str(err.value)

    tst = deepcopy(data)
    tst.Sample1 = "a"
    with pytest.raises(AssertionError) as err:
        ldata(tst, desc, annot)
    assert "data must all be numeric values" in str(err.value)

    x = ldata(data, desc, annot)

    assert isinstance(x, ldata)
    assert x.data.equals(data)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)

    snapshot.assert_match(str(x), "ldata.txt")


def test_data(snapshot):
    x = ldata(data, desc, annot)
    new_data = deepcopy(data) + 1
    with pytest.raises(AssertionError) as err:
        x.data = new_data.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    x.data = new_data
    assert x.data.equals(new_data)
    snapshot.assert_match(x.data.to_csv(), "data.txt")


def test_description(snapshot):
    x = ldata(data, desc, annot)
    new_desc = deepcopy(desc)
    new_desc["Group"] = ["Group1", "Group1", "Group1", "Group2", "Group2"]
    with pytest.raises(AssertionError) as err:
        x.description = new_desc.head(2)
    assert "data colnames do not match description ID" in str(err.value)

    x.description = new_desc
    assert x.description.equals(new_desc)
    snapshot.assert_match(x.description.to_csv(), "description.txt")


def test_annotation(snapshot):
    x = ldata(data, desc, annot)
    new_annot = deepcopy(annot)
    new_annot["Group"] = ["Group" + str(i) for i in range(1, 11)] + [
        "Group" + str(i) for i in range(1, 11)
    ]
    with pytest.raises(AssertionError) as err:
        x.annotation = new_annot.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    x.annotation = new_annot
    assert x.annotation.equals(new_annot)
    snapshot.assert_match(x.annotation.to_csv(), "annotation.txt")


def test_dimnames():
    x = ldata(data, desc, annot)

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

    x.dimnames = [x.rownames, ["A", "B", "C", "D", "E"][::-1]]
    assert x.colnames == ["A", "B", "C", "D", "E"][::-1]
    assert x.description["ID"].tolist() == ["A", "B", "C", "D", "E"][::-1]


def test_example_ldata(snapshot):
    x = ldata.example_ldata()
    assert isinstance(x, ldata)
    snapshot.assert_match(str(x), "example_ldata.txt")


def test_subset(snapshot):
    x = ldata(data, desc, annot)

    with pytest.raises(AssertionError) as err:
        x.subset(samples=["sample"])
    assert "samples are not in data" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.subset(features=["feature"])
    assert "features are not in data" in str(err.value)

    samples = ["Sample1", "Sample3"]
    features = ["Feature1", "Feature3", "Feature18"]
    s = x.subset(samples=samples, features=features)

    assert isinstance(s, ldata)
    assert s.colnames == samples
    assert s.rownames == features
    assert s.description["ID"].tolist() == samples
    assert s.description.index.tolist() == [0, 1]
    assert s.annotation["ID"].tolist() == features
    assert s.annotation.index.tolist() == [0, 1, 2]
    assert s.data.columns.tolist() == samples
    assert s.data.index.tolist() == features
    assert x.colnames == data.columns.tolist()
    assert x.rownames == data.index.tolist()
    snapshot.assert_match(str(s), "subset_ldata.txt")


def test_transpose(snapshot):
    x = ldata(data, desc, annot)

    l = x.transpose()

    assert isinstance(l, ldata)
    assert l.colnames == x.rownames
    assert l.rownames == x.colnames
    assert l.description["ID"].tolist() == x.annotation["ID"].tolist()
    assert l.annotation["ID"].tolist() == x.description["ID"].tolist()
    assert l.data.columns.tolist() == x.data.index.tolist()
    assert l.data.index.tolist() == x.data.columns.tolist()
    assert x.colnames == data.columns.tolist()
    assert x.rownames == data.index.tolist()
    snapshot.assert_match(str(l), "transpose_ldata.txt")


def test_concat(snapshot):
    a = ldata.example_ldata(type="simulate")
    b = ldata.example_ldata(type="simulate", min=5, max=6)
    c = ldata.example_ldata(type="simulate", min=15, max=20)

    with pytest.raises(AssertionError) as err:
        a.concat([b, "c"])
    assert "objects must all be of same class" in str(err.value)

    b.rownames = ["Feature" + str(i) for i in range(51, 71)]
    with pytest.raises(AssertionError) as err:
        a.concat([b, c])
    assert "objects must have same feature IDs" in str(err.value)
    b.rownames = c.rownames

    with pytest.raises(AssertionError) as err:
        a.concat([b, c])
    assert "colnames must contain unique IDs" in str(err.value)
    b.colnames = ["Sample" + str(i) for i in range(6, 11)]
    c.colnames = ["Sample" + str(i) for i in range(11, 16)]

    x = a.concat([b, c])

    assert x.data.equals(pd.concat([a.data, b.data, c.data], axis=1))
    assert x.annotation.equals(a.annotation)
    assert x.description.equals(
        pd.concat([a.description, b.description, c.description]).reset_index(drop=True)
    )
    snapshot.assert_match(str(x), "concat_ldata.txt")
