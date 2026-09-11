"""
DIPK on the stock gene list (2,270 genes) with log2(TPM+1) gene expression.

This isolates the SCALE effect in the small gene space: it differs from stock
:class:`~drevalpy.models.DIPK.dipk.DIPKModel` in exactly one respect — the gene expression is log
transformed before it reaches the gene autoencoder. Gene list, architecture and hyperparameters are
untouched.

Why: drevalpy hands the dataset files to DIPK unchanged, and CTRPv2 gene expression is LINEAR TPM,
while the DIPK publication always feeds log scale expression into the autoencoder (GDSC RMA is log2
by construction, the CCLE branch applies ``np.log2(ft + 1)`` explicitly). On the 11,883 gene space
the correction turned out to be a wash (``gp20260803_dipklog_resid`` nSpear 0.313 vs.
``gp20260801_dipk11883_resid`` 0.327); this model checks whether that also holds on the 2,270 gene
space that all earlier DIPK runs used.
"""

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.utils import load_and_select_gene_features

from ..DIPK.data_utils import load_bionic_features
from ..DIPK.dipk import DIPKModel
from ..DIPKLOG.dipklog import log_transform

GENE_LIST = "gene_expression_intersection"


class DIPKLOG2270Model(DIPKModel):
    """DIPK on the stock 2,270 gene intersection with log2(TPM+1) expression."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: DIPKLOG2270
        """
        return "DIPKLOG2270"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features on the stock gene list, log transformed.

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
