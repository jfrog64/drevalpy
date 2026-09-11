"""
DIPK on the TCGA autoencoder gene space (11,883 genes) with log2(TPM+1) gene expression.

Control model for Aufgabe B, stage 2. It differs from :class:`DIPK11883Model` in exactly one
respect: the gene expression is log transformed before it reaches the gene autoencoder.

Why: the DIPK publication always feeds log scale expression into the autoencoder — the GDSC branch
uses RMA processed microarray data (log2 by construction) and the CCLE branch applies
``np.log2(ft + 1)`` explicitly (Task3/DataPreprocess/TPM/RNAseq_Dict.py). drevalpy passes through
whatever the dataset files contain, and those are inconsistent: GDSC1 gene expression is on log
scale, CTRPv2 is LINEAR TPM. On linear TPM the top 100 genes carry ~80% of the total variance, so
an MSE autoencoder reconstructs housekeepers and ignores the remaining 11,783 genes; on
log2(TPM+1) the top 100 genes carry ~6%.

This model is the "encoder from scratch, correct scale" control against which the TCGA pretrained
:class:`~drevalpy.models.DIPKTCGA.dipktcga.DIPKTCGAModel` is compared.
"""

import numpy as np

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.utils import load_and_select_gene_features

from ..DIPK.data_utils import load_bionic_features
from ..DIPK.dipk import DIPKModel

GENE_LIST = "tcga_autoencoder_genes_available"


def log_transform(gene_expression: FeatureDataset) -> None:
    """
    Transform linear TPM gene expression to log2(TPM+1) in place.

    :param gene_expression: feature dataset holding a ``gene_expression`` view in linear TPM
    """
    gene_expression.apply(lambda x: np.log2(np.clip(x, 0.0, None) + 1.0), view="gene_expression")


class DIPKLOGModel(DIPKModel):
    """DIPK on 11,883 genes with log2(TPM+1) expression, autoencoder trained from scratch."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: DIPKLOG
        """
        return "DIPKLOG"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features on the TCGA autoencoder gene space, log transformed.

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
        log_transform(gene_expression)
        bionic_features = load_bionic_features(
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features.add_features(gene_expression)

        return bionic_features
