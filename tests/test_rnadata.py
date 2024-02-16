import pytest
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from pydata.rnadata import rnadata
from rnanorm.datasets import load_toy_data
import numpy as np
import pandas as pd

dat = load_toy_data()

data = dat.exp.transpose()
desc = pd.DataFrame({"ID": dat.exp.index})
annot = pd.DataFrame({"ID": dat.exp.columns})
gtf = dat.gtf_path

def test_rnadata_generation(snapshot):
    with pytest.raises(AssertionError) as err:
        rnadata(data.head(2), desc, annot, gtf)
    assert "data rownames do not match annotation ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        rnadata(data, desc.head(2), annot, gtf)
    assert "data colnames do not match description ID" in str(err.value)

    with pytest.raises(AssertionError) as err:
        rnadata(data, desc, annot.head(2), gtf)
    assert "data rownames do not match annotation ID" in str(err.value)

    x = rnadata(data, desc, annot, gtf)

    assert isinstance(x, rnadata)
    assert x.data.equals(data)
    assert x.description.equals(desc)
    assert x.annotation.equals(annot)

    snapshot.assert_match(str(x), "rnadata.txt")


def test_example_rnadata(snapshot):
    x = rnadata.example_rnadata()
    assert isinstance(x, rnadata)
    snapshot.assert_match(str(x), "example_rnadata.txt")