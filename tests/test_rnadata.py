import pytest
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from pydata.rnadata import rnadata

from rnanorm.datasets import load_toy_data
import numpy as np
import pandas as pd
import os

dat = load_toy_data()

data = dat.exp.transpose()
desc = pd.DataFrame({"ID": dat.exp.index})
annot = pd.DataFrame({"ID": dat.exp.columns})
gtf = os.path.basename(str(dat.gtf_path))


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
    x.gtf = gtf
    assert isinstance(x, rnadata)
    snapshot.assert_match(str(x), "example_rnadata.txt")


def test_rnadata_count_filtering():
    x = rnadata.example_rnadata("gtex")
    with pytest.raises(Exception) as err:
        x.filter_counts(method="custom")
    assert "custom filtering not implemented" in str(err.value)

    out = x.filter_counts()
    keep = x.data.sum(axis=1) >= 10
    assert keep[keep].index.tolist() == out.rownames

    out = x.filter_counts(method="mean")
    keep = x.data.mean(axis=1) >= 10
    assert keep[keep].index.tolist() == out.rownames

    out = x.filter_counts(method="min")
    keep = x.data.min(axis=1) >= 10
    assert keep[keep].index.tolist() == out.rownames


def test_rnadata_normalisation(snapshot):
    x = rnadata.example_rnadata()
    with pytest.raises(Exception) as err:
        x.normalise(method="custom")
    assert "custom normalisation not implemented" in str(err.value)
    for norm in ["CPM", "TPM", "FPKM", "UQ", "CUF", "TMM", "CTF"]:
        norm_x = x.normalise(method=norm)
        assert isinstance(norm_x, rnadata)
        assert norm_x.normalisation_method == norm
