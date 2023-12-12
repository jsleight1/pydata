import pytest
from pydata.pca import pca
from copy import deepcopy
import numpy as np 
import pandas as pd

np.random.seed(38)
pcs = pd.DataFrame(
    np.random.randn(5, 5),
    index = ["PC" + str(i)for i in range(1, 6)], 
    columns = ["Sample" + str(i)for i in range(1, 6)] 
)
desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
annot = pd.DataFrame(
    {
        "ID": ["PC" + str(i) for i in range(1, 6)],
        "Percentage variance explained": [52.5, 32.5, 8.1, 5.2, 1.7]
    }
)

def test_pca_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        pca(pcs.head(2), desc, annot)
    assert "data rownames do not match annotation ID" in str(err.value)
    
    with pytest.raises(AssertionError) as err:
        pca(pcs, desc.head(2), annot)
    assert "data colnames do not match description ID" in str(err.value)
    
    with pytest.raises(AssertionError) as err:
        pca(pcs, desc, annot.head(2))
    assert "data rownames do not match annotation ID" in str(err.value)
    
    x = pca(pcs, desc, annot)

    assert isinstance(x, pca)
    assert x.data.equals(pcs)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)
    
    snapshot.assert_match(str(x), "pca.txt")

def test_annotation(snapshot):
    x = pca(pcs, desc, annot)
    new_annot = deepcopy(annot)
    new_annot["ID"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(AssertionError) as err:
        x.annotation = new_annot
    assert "ID column must be in format PC1, PC2, etc" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.annotation = annot.drop(["Percentage variance explained"], axis = 1)
    assert "annotation must contain 'Percentage variance explained' column" in str(err.value)

    with pytest.raises(AssertionError) as err:
        x.annotation = annot.head(2)
    assert "data rownames do not match annotation ID" in str(err.value)

    new_annot = deepcopy(annot)
    new_annot["col"] = ["a", "b", "c", "d", "e"]
    x.annotation = new_annot
    assert x.annotation.equals(new_annot)
    snapshot.assert_match(x.annotation.to_csv(), "annotation.txt")

def test_dimnames():
    x = pca(pcs, desc, annot)

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
    assert "rownames must be in format PC1, PC2, etc" in str(err.value)

    x.dimnames = [x.rownames, ["A", "B", "C", "D", "E"][::-1]]
    assert x.colnames == ["A", "B", "C", "D", "E"][::-1]
    assert x.description["ID"].tolist() == ["A", "B", "C", "D", "E"][::-1]

def test_subset():
    x = pca(pcs, desc, annot)
    with pytest.raises(Exception) as err:
        x.subset()
    assert "Cannot subset pca object" in str(err.value)

def test_transpose():
    x = pca(pcs, desc, annot)
    with pytest.raises(Exception) as err:
        x.transpose()
    assert "Cannot transpose pca object" in str(err.value)