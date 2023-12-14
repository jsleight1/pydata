from pydata.ldata import ldata
import re


class drdata(ldata):
    """
    Class to store results from a dimension reduction analysis
    """

    def __init__(self, data, description, annotation, target=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame of reduce dimensions for ncol samples and nrow
            components.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation : pandas.DataFrame
            A DataFrame of reduce dimensions annotation
        """

        super().__init__(data, description, annotation)

    def subset(self):
        t = re.findall("'([^']*)'", str(type(self)))[0].split(".")[-1]
        raise Exception(f"Cannot subset {t} object")

    def transpose(self):
        t = re.findall("'([^']*)'", str(type(self)))[0].split(".")[-1]
        raise Exception(f"Cannot transpose {t} object")

    def concat(self, objs=[]):
        t = re.findall("'([^']*)'", str(type(self)))[0].split(".")[-1]
        raise Exception(f"Cannot concat {t} object")
