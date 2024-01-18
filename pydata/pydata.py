from pydata.ldata import ldata
from pydata.pca import pca
from pydata.lda import lda
from pydata.tsne import tsne
from pydata.umap import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import seaborn as sns
import plotly.express as px
from copy import deepcopy


class pydata(ldata):
    """
    Simple dataset computation and visualisation.

    pydata objects use a L-shaped data structure where the underlying
    data set is a DataFrame of numeric values with an associated
    description and annotation DataFrame containing relevant sample
    and feature metadata respectively.

    Examples
    --------

    >>> # Generate some random data
    >>> np.random.seed(38)
    >>> data = pd.DataFrame(
    >>>     np.random.randint(0, 10, size = 100).reshape(20, 5),
    >>>     index = ["Feature" + str(i)for i in range(1, 21)],
    >>>     columns = ["Sample" + str(i)for i in range(1, 6)]
    >>> )
    >>>
    >>> # Generate sample metadata
    >>> desc = pd.DataFrame({"ID": ["Sample" + str(i) for i in range(1, 6)]})
    >>>
    >>> # Generate feature metadata
    >>> annot = pd.DataFrame({"ID": ["Feature" + str(i) for i in range(1, 21)]})
    >>>
    >>> # Generate pydata object
    >>> x = pydata(data, desc, annot)
    >>> x
    >>> x.data
    >>> x.description
    >>> x.annotation
    >>> x.perform_dimension_reduction("pca")
    >>> x.plot("pca")
    >>> x.plot("distribution")
    """

    def __init__(self, data, description, annotation):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame of data with columns representing samples and rows
            representing features.
        description: pandas.DataFrame
            A DataFrame of sample descriptions with ID column matching
            columns names of data attribute.
        annotation: pandas.DataFrame
            A DataFrame of feature annotation with ID column matching
            row names of data attribute.
        """

        super().__init__(data, description, annotation)

    @staticmethod
    def example_pydata(**kwargs):
        """
        Generate example pydata. See ldata.example_ldata for details.
        """
        out = ldata.example_ldata(**kwargs)
        return pydata(out.data, out.description, out.annotation)

    def _get_pcs(self):
        return getattr(self, "_pcs")

    def _set_pcs(self, value: pca):
        if value is not None:
            assert isinstance(value, pca)
            value._validate()
        self._pcs = value

    pcs = property(_get_pcs, _set_pcs)

    def _get_lda(self):
        return getattr(self, "_lda")

    def _set_lda(self, value: lda):
        if value is not None:
            assert isinstance(value, lda)
            value._validate()
        self._lda = value

    lda = property(_get_lda, _set_lda)

    def _get_tsne(self):
        return getattr(self, "_tsne")

    def _set_tsne(self, value: tsne):
        if value is not None:
            assert isinstance(value, tsne)
            value._validate()
        self._tsne = value

    tsne = property(_get_tsne, _set_tsne)

    def _get_umap(self):
        return getattr(self, "_umap")

    def _set_umap(self, value: umap):
        if value is not None:
            assert isinstance(value, umap)
            value._validate()
        self._umap = value

    umap = property(_get_umap, _set_umap)

    def subset(self, samples=None, features=None):
        out = super().subset(samples=samples, features=features)
        out.pcs = None
        out.lda = None
        out.tsne = None
        out.umap = None
        return out

    def transpose(self):
        out = super().transpose()
        out.pcs = None
        out.lda = None
        out.tsne = None
        out.umap = None
        return out

    def concat(self, objs=[]):
        out = super().concat(objs=objs)
        out.pcs = None
        out.lda = None
        out.tsne = None
        out.umap = None
        return out

    def perform_dimension_reduction(self, type: str, **kwargs):
        """
        Perform dimension reduction.
        ---------------------------
        type: str
            Type of dimension reduction to perform. Either "pca" for principal
            component analysis, "lda" for linear discriminant analysis, "tsne"
            for t-distributed stochastic neighbor embedding or "umap" for
            uniform manifold approximation and projection.
        **kwargs:
            Passed to dimension reduction methods.
        """
        self._validate()
        match type:
            case "pca":
                self.pcs = pca.analyse(self, **kwargs)
            case "lda":
                self.lda = lda.analyse(self, **kwargs)
            case "tsne":
                self.tsne = tsne.analyse(self, **kwargs)
            case "umap":
                self.umap = umap.analyse(self, **kwargs)
            case _:
                raise Exception(type + " dimension reduction not implemented")

    def plot(self, type: str, **kwargs):
        """
        Plot pydata object.
        ------------------
        type: str
            Type of plot. Either "pca", "pca_elbow", "lda", "tsne", "umap",
            "violin", "feature_heatmap", "correlation_heatmap", "distribution"
            or "scatter"
        **kwargs:
            Passed to plotting methods
        """
        self._validate()
        match type:
            case "pca":
                self.pcs.plot(type="scatter", **kwargs)
            case "pca_elbow":
                self.pcs.plot(type="elbow", **kwargs)
            case "lda":
                self.lda.plot(**kwargs)
            case "tsne":
                self.tsne.plot(**kwargs)
            case "umap":
                self.umap.plot(**kwargs)
            case "violin":
                self._violin_plot(**kwargs)
            case "feature_heatmap":
                self._feature_heatmap(**kwargs)
            case "correlation_heatmap":
                self._correlation_heatmap(**kwargs)
            case "distribution":
                self._distribution_plot(**kwargs)
            case "scatter":
                self._scatter_plot(**kwargs)
            case _:
                raise Exception(type + " plot type not implemented")

    def _plot_data(self):
        df = deepcopy(self.data)
        df["Feature"] = df.index
        df = pd.melt(df, id_vars="Feature", value_vars=self.colnames, var_name="Sample")
        df = pd.merge(df, self.description, left_on="Sample", right_on="ID")
        df = df.drop(columns=["ID"])
        df = pd.merge(df, self.annotation, left_on="Feature", right_on="ID")
        df = df.drop(columns=["ID"])
        return df

    def _violin_plot(self, interactive: bool=False, **kwargs):
        if interactive:
            px.violin(data_frame=self._plot_data(), x="Sample", y="value", **kwargs).show()
        else:
            sns.violinplot(data=self._plot_data(), x="Sample", y="value", **kwargs)
            plt.xticks(rotation=90)
            plt.show()

    def _correlation_heatmap(
        self,
        cor_method: str = "pearson",
        annotate_samples_by=None,
        cmap="coolwarm",
        **kwargs,
    ):
        dat = self.data.corr(method=cor_method)
        if annotate_samples_by is not None:
            annotate_samples_by = self._colour_by_df(
                self.description, annotate_samples_by
            )
            annotate_samples_by["colour_df"].index = self.description["ID"]

        self._heatmap(
            dat,
            annotate_samples_by=annotate_samples_by,
            cmap=cmap,
            cbar_kws={"label": cor_method.capitalize() + " correlation"},
            **kwargs,
        )

    def _feature_heatmap(
        self,
        annotate_samples_by=None,
        annotate_features_by=None,
        cmap="coolwarm",
        **kwargs,
    ):
        if annotate_samples_by is not None:
            annotate_samples_by = self._colour_by_df(
                self.description, annotate_samples_by
            )
            annotate_samples_by["colour_df"].index = self.description["ID"]

        if annotate_features_by is not None:
            annotate_features_by = self._colour_by_df(
                self.annotation, annotate_features_by
            )
            annotate_features_by["colour_df"].index = self.annotation["ID"]

        self._heatmap(
            self.data,
            cbar_kws={"label": "Feature value"},
            annotate_samples_by=annotate_samples_by,
            annotate_features_by=annotate_features_by,
            cmap=cmap,
            **kwargs,
        )

    @staticmethod
    def _colour_by_df(x, colour_by):
        colours = []
        lut = []
        for i in colour_by:
            col = sns.color_palette("Spectral", n_colors=len(x[i].unique()))
            lut += [dict(zip(x[i].unique(), col))]
            colours += [x[i].map(dict(zip(x[i].unique(), col)))]
        colours = pd.concat(colours, axis=1)
        return {"colour_df": colours, "colour_dict": lut}

    def _heatmap(
        self, x, annotate_samples_by=None, annotate_features_by=None, **kwargs
    ):
        if annotate_samples_by is not None:
            assert (
                self.data.columns.tolist()
                == annotate_samples_by["colour_df"].index.tolist()
            ), "annotate_samples_by index must be same as data columns"
        else:
            annotate_samples_by = {"colour_df": None}

        if annotate_features_by is not None:
            assert (
                self.data.index.tolist()
                == annotate_features_by["colour_df"].index.tolist()
            ), "annotate_features_by index must be same as data columns"
        else:
            annotate_features_by = {"colour_df": None}

        plot = sns.clustermap(
            x,
            col_colors=annotate_samples_by["colour_df"],
            row_colors=annotate_features_by["colour_df"],
            dendrogram_ratio=0.2,
            **kwargs,
        )

        # Add legends
        if annotate_samples_by["colour_df"] is not None:
            for i in range(0, len(annotate_samples_by["colour_dict"])):
                lut = annotate_samples_by["colour_dict"][i]
                for key in lut:
                    plot.ax_col_dendrogram.bar(
                        0, 0, color=lut[key], label=key, linewidth=0
                    )
                plot.ax_col_dendrogram.legend(title="Samples", loc="center", ncol=3)

        if annotate_features_by["colour_df"] is not None:
            for i in range(0, len(annotate_features_by["colour_dict"])):
                lut = annotate_features_by["colour_dict"][i]
                for key in lut:
                    plot.ax_row_dendrogram.bar(
                        0, 0, color=lut[key], label=key, linewidth=0
                    )
                plot.ax_row_dendrogram.legend(
                    title="Features", loc="center left", ncol=1
                )

        plot.fig.subplots_adjust(right=0.7)
        plot.ax_cbar.set_position((0.8, 0.2, 0.03, 0.4))
        plt.show()

    def _distribution_plot(self, **kwargs):
        sns.displot(data=self.data, **kwargs)
        plt.xlabel("Feature value")
        plt.show()

    def _scatter_plot(self, interactive: bool=False, xaxis=None, yaxis=None, **kwargs):
        if xaxis is None:
            xaxis = self.colnames[0]
        if yaxis is None:
            yaxis = self.colnames[1]
        df = deepcopy(self.data.reset_index(names="ID"))
        df = pd.merge(df, self.annotation, on="ID")
        if interactive:
            px.scatter(
                df, x=xaxis, y=yaxis, hover_name="ID", **kwargs
            ).show()
        else:
            sns.regplot(data=df, x=xaxis, y=yaxis, seed=32, **kwargs)
            plt.show()
