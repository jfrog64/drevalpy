"""
DIPK with a gene autoencoder pretrained on TCGA tumors (Aufgabe B, stage 2).

Identical to :class:`~drevalpy.models.DIPKLOG.dipklog.DIPKLOGModel` — same 11,883 gene space, same
log2(TPM+1) scale — except for how the gene encoder is obtained: instead of training the denoising
autoencoder from random initialization on the cell lines of the current fold, the weights
pretrained unsupervised on 10,535 TCGA tumors (stage 1, ``pretrain_tcga_autoencoder_20260803.py``)
are loaded and then fine tuned on the fold's training cell lines.

The two models therefore differ ONLY in the initialization of the gene autoencoder, which makes
DIPKLOG the clean control for the effect of the TCGA pretraining.
"""

import os

import numpy as np
import torch

from ..DIPK.gene_expression_encoder import GeneExpressionEncoder, train_gene_expession_autoencoder
from ..DIPKLOG.dipklog import DIPKLOGModel


class DIPKTCGAModel(DIPKLOGModel):
    """DIPK on 11,883 log2(TPM+1) genes with a TCGA pretrained, fine tuned gene autoencoder."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: DIPKTCGA
        """
        return "DIPKTCGA"

    def _fit_gene_encoder(
        self, train_gene_expression: np.ndarray, val_gene_expression: np.ndarray
    ) -> GeneExpressionEncoder:
        """
        Warm start the gene autoencoder from the TCGA pretrained weights and fine tune it.

        :param train_gene_expression: gene expression of the training rows
        :param val_gene_expression: gene expression of the early stopping rows
        :returns: fine tuned gene expression encoder
        :raises FileNotFoundError: if the pretrained weights are missing
        :raises ValueError: if the pretrained weights do not match the current gene space
        """
        directory = self.hyperparameters["pretrained_encoder_dir"]
        encoder_path = os.path.join(directory, "gene_encoder.pt")
        decoder_path = os.path.join(directory, "gene_decoder.pt")
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            raise FileNotFoundError(
                f"DIPKTCGA needs pretrained autoencoder weights in {directory} "
                "(run pretrain_tcga_autoencoder_20260803.py first)."
            )

        encoder_state = torch.load(encoder_path, map_location="cpu")  # noqa: S614
        decoder_state = torch.load(decoder_path, map_location="cpu")  # noqa: S614
        pretrained_input_dim = encoder_state["encoder.0.0.weight"].shape[1]
        if pretrained_input_dim != train_gene_expression.shape[1]:
            raise ValueError(
                f"Pretrained encoder expects {pretrained_input_dim} genes, but the data has "
                f"{train_gene_expression.shape[1]}. Gene list and pretraining must match."
            )

        print(f"DIPKTCGA: fine tuning the TCGA pretrained gene autoencoder from {directory}")
        return train_gene_expession_autoencoder(
            train_gene_expression,
            val_gene_expression,
            epochs_autoencoder=self.hyperparameters["epochs_autoencoder"],
            encoder_state=encoder_state,
            decoder_state=decoder_state,
            lr=self.hyperparameters.get("lr_autoencoder_finetune", 1e-5),
        )
