import pytest
from pydata.lda import lda
from copy import deepcopy
import numpy as np 
import pandas as pd

np.random.seed(38)
lds = pd.DataFrame(
    np.random.randn(5, 5),
    index = ["LD" + str(i)for i in range(1, 6)], 
    columns = ["Sample" + str(i)for i in range(1, 6)] 
)
desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
annot = pd.DataFrame({"ID": ["LD" + str(i) for i in range(1, 6)]})

def test_lda_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        lda(lds.head(2), desc, annot)
    assert "data rownames do not match annotation ID" in str(err.value)
    
    with pytest.raises(AssertionError) as err:
        lda(lds, desc.head(2), annot)
    assert "data colnames do not match description ID" in str(err.value)
    
    with pytest.raises(AssertionError) as err:
        lda(lds, desc, annot.head(2))
    assert "data rownames do not match annotation ID" in str(err.value)
    
    x = lda(lds, desc, annot)

    assert isinstance(x, lda)
    assert x.data.equals(lds)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)
    
    snapshot.assert_match(str(x), "lda.txt")

def test_annotation(snapshot):
    x = lda(lds, desc, annot)
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
    x = lda(lds, desc, annot)

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
    assert "rownames must be in format LD1, LD2, etc" in str(err.value)

    x.dimnames = [x.rownames, ["A", "B", "C", "D", "E"][::-1]]
    assert x.colnames == ["A", "B", "C", "D", "E"][::-1]
    assert x.description["ID"].tolist() == ["A", "B", "C", "D", "E"][::-1]