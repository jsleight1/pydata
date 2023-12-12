import pytest
from pydata.pydata import pydata
from pydata.pca import pca
from pydata.lda import lda
import numpy as np 
import pandas as pd

np.random.seed(38)
data = pd.DataFrame(
    np.random.randint(0, 10, size = 120).reshape(20, 6),
    index = ["Feature" + str(i)for i in range(1, 21)], 
    columns = ["Sample" + str(i)for i in range(1, 7)] 
)
grps = np.array(["Control", "Treatment"])
desc = pd.DataFrame(
    {
        "ID": ["Sample" + str(i) for i in range(1, 7)], 
        "Treatment": np.repeat(grps, [3, 3], axis = 0)
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

def test_perform_pca(snapshot):
    x = pydata(data, desc, annot)

    with pytest.raises(Exception) as err:
        x.perform_pca(method = "custom")
    assert "custom pca method not implemented" in str(err.value)

    with pytest.raises(Exception) as err:
        x.perform_pca(scaling = "custom")
    assert "custom scaling method not implemented" in str(err.value)

    x.perform_pca()

    assert isinstance(x.pcs, pca)
    snapshot.assert_match(str(x.pcs), "pydata_pca_print.txt")
    snapshot.assert_match(x.pcs.data.round(3).to_csv(), "pydata_pca_data.txt")

def test_perform_lda(snapshot):
    x = pydata(data, desc, annot)

    with pytest.raises(Exception) as err:
        x.perform_lda(target = "group", n_comp = 1)
    assert "group is not in pydata description" in str(err.value)

    x.perform_lda(target = "Treatment", n_comp = 1)

    assert isinstance(x.lda, lda)
    snapshot.assert_match(str(x.lda), "pydata_lda_print.txt")
    snapshot.assert_match(x.lda.data.round(3).to_csv(), "pydata_lda_data.txt")