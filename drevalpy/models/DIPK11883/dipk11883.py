"""
DIPK on the TCGA autoencoder gene space (11,883 genes).

Identical to :class:`~drevalpy.models.DIPK.dipk.DIPKModel` except for the gene list behind the
gene expression view: stock DIPK uses ``gene_expression_intersection`` (2,270 genes), this variant
uses ``tcga_autoencoder_genes_available`` (11,883 genes = the autoencoder gene list restricted to
the symbols that have a gencode.v23 locus in the TCGA matrix; only SCAND3 is dropped).

Purpose: the CTRPv2 baseline has to live in the SAME feature space as the TCGA-pretrained encoder
of Aufgabe B, otherwise a comparison "encoder from scratch" vs. "encoder pretrained on TCGA" would
confound pretraining with the gene set.
"""

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.utils import load_and_select_gene_features

from ..DIPK.data_utils import load_bionic_features
from ..DIPK.dipk import DIPKModel

GENE_LIST = "tcga_autoencoder_genes_available"


class DIPK11883Model(DIPKModel):
    """DIPK trained on the 11,883-gene TCGA autoencoder gene space."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: DIPK11883
        """
        return "DIPK11883"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features on the TCGA autoencoder gene space instead of the intersection list.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: cell line features
        """
        gene_expression = load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list=GENE_LIST,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features = load_bionic_features(
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features.add_features(gene_expression)

        return bionic_features
