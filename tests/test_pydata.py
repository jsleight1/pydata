import pytest
from pydata.pydata import pydata
from copy import deepcopy
import numpy as np 
import pandas as pd

np.random.seed(38)
data = pd.DataFrame(
    np.random.randint(0, 10, size=100).reshape(20, 5),
    index=["Feature" + str(i)for i in range(1, 21)], 
    columns=["Sample" + str(i)for i in range(1, 6)] 
)
desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
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


def test_data(snapshot):
    x = pydata(data, desc, annot)
    new_data = deepcopy(data) + 1
    with pytest.raises(AssertionError) as err:
        x.data = new_data.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    x.data = new_data
    assert x.data.equals(new_data)
    snapshot.assert_match(x.data.to_csv(), "data.txt")


def test_description(snapshot):
    x = pydata(data, desc, annot)
    new_desc = deepcopy(desc)
    new_desc["Group"] = ["Group1", "Group1", "Group1", "Group2", "Group2"]
    with pytest.raises(AssertionError) as err:
        x.description = new_desc.head(2)
    assert "data colnames do not match description ID" in str(err.value)

    x.description = new_desc
    assert x.description.equals(new_desc)
    snapshot.assert_match(x.description.to_csv(), "description.txt")

def test_annotation(snapshot):
    x = pydata(data, desc, annot)
    new_annot = deepcopy(annot)
    new_annot["Group"] = ["Group" + str(i)for i in range(1, 11)] + \
        ["Group" + str(i)for i in range(1, 11)]
    with pytest.raises(AssertionError) as err:
        x.annotation = new_annot.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    x.annotation = new_annot
    assert x.annotation.equals(new_annot)
    snapshot.assert_match(x.annotation.to_csv(), "annotation.txt")

def test_dimnames(snapshot):
    x = pydata(data, desc, annot)

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