"""
ElasticNet-Varianten mit DRUG-MEAN-RESIDUALISIERUNG *in der Architektur*.

Hintergrund (DrEval, 2026-07): Die Drug-Mean-Residualisierung (Drug-Mittelwert des
Train-Folds vom Trainingsziel abziehen, bei predict wieder addieren) hob das lineare
ElasticNet in der normalisierten Metrik stark (EN norm. Spearman ~0 -> 0.147), während
sie dem RandomForest schadete (-0.066) und Precily/SNN unterschiedlich reagierten. Bisher
war der Effekt nur über den *globalen* Schalter ``--response_transformation drug_mean``
verfügbar, der einen ganzen Lauf residualisiert und daher resid- und non-resid-Modelle
NICHT im selben Ranking nebeneinander stellen kann.

Dieses Modul baut die Residualisierung als **Modell-Eigenschaft** ein (``_DrugResidMixin``):
``train`` schätzt den Drug-Mittelwert auf dem Train-Fold, zieht ihn vom Ziel ab und
trainiert auf den Residuen; ``predict`` addiert den Drug-Mittelwert wieder auf (unbekannte
Drugs -> globaler Train-Mittelwert). Der Mixin ist mit dem ``_TargetMutMixin`` komponierbar,
sodass sich das volle 2x2x2-Faktorial {270,893} x {Mut, kein Mut} x {resi, kein resi}
in EINEM drevalpy-Report ranken lässt.

Der Residualisierer selbst (``GroupMeanCenterer``) ist derselbe, der auch die früheren
resid-Läufe (RF/SNN/Precily) getragen hat -> identische, bereits validierte Mechanik.

Nicht upstream-tauglich (hängt an den DrEval-Playground-Konventionen) -- Playground-Klon.
"""

import os

import joblib
import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ...response_transformation import GroupMeanCenterer
from .sklearn_models import ElasticNetModel
from .target_mut_sklearn import (
    ElasticNetTargetMut,
    ElasticNetTargetMut893,
    _GeneListMixin,
)


class _DrugResidMixin:
    """Mixin: Drug-Mean-Residualisierung des Trainingsziels, im Modell gekapselt.

    MRO-Hinweis: dieser Mixin MUSS vor der Basisklasse (und vor ``_TargetMutMixin``) stehen,
    damit ``train`` das Ziel residualisiert, bevor die Basis darauf fittet, und ``predict``
    den Offset nach der Basis-Vorhersage wieder addiert.
    """

    # Klassen-Default: existiert auch, bevor train lief bzw. wenn load() ohne State-Datei lädt.
    _resid_centerer: GroupMeanCenterer | None = None
    _RESID_STATE_FILE = "drug_resid_state.pkl"

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Drug-Mean auf dem Train-Fold schätzen, Ziel residualisieren, dann Basis-Training.

        Der Drug-Mittelwert wird NUR aus ``output`` (dem Train-Fold) geschätzt -> kein Leakage.
        ``output._response`` wird temporär auf die Residuen gesetzt und im ``finally`` wieder
        auf das Original zurückgestellt (die Response ist ein Property ohne Setter; der interne
        drevalpy-Code schreibt ebenfalls direkt auf ``_response``).
        """
        self._resid_centerer = GroupMeanCenterer()
        if len(output) == 0:
            super().train(
                output=output, cell_line_input=cell_line_input, drug_input=drug_input,
                output_earlystopping=output_earlystopping, model_checkpoint_dir=model_checkpoint_dir,
            )
            return

        self._resid_centerer.fit(output.response, groups=output.drug_ids)
        orig = output.response
        es_orig = None
        try:
            output._response = self._resid_centerer.transform(output.response, groups=output.drug_ids)
            if output_earlystopping is not None and len(output_earlystopping) > 0:
                es_orig = output_earlystopping.response
                output_earlystopping._response = self._resid_centerer.transform(
                    output_earlystopping.response, groups=output_earlystopping.drug_ids
                )
            super().train(
                output=output, cell_line_input=cell_line_input, drug_input=drug_input,
                output_earlystopping=output_earlystopping, model_checkpoint_dir=model_checkpoint_dir,
            )
        finally:
            output._response = orig
            if es_orig is not None:
                output_earlystopping._response = es_orig

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Basis-Vorhersage (auf Residual-Skala) + Drug-Mittelwert zurückaddieren -> Originalskala."""
        resid_pred = super().predict(
            cell_line_ids=cell_line_ids, drug_ids=drug_ids,
            cell_line_input=cell_line_input, drug_input=drug_input,
        )
        if self._resid_centerer is None:
            return resid_pred
        return self._resid_centerer.inverse_transform(resid_pred, groups=drug_ids)

    def save(self, directory: str) -> None:
        """Basis-Artefakte + den gefitteten Residualisierer speichern.

        :param directory: Zielverzeichnis
        """
        super().save(directory)
        joblib.dump(self._resid_centerer, os.path.join(directory, self._RESID_STATE_FILE))

    @classmethod
    def load(cls, directory: str):
        """Basis-`load` + Wiederherstellung des Residualisierers.

        :param directory: Verzeichnis mit den gespeicherten Modell-Artefakten
        :returns: wiederhergestellte Modellinstanz
        """
        instance = super().load(directory)
        state_path = os.path.join(directory, cls._RESID_STATE_FILE)
        if os.path.exists(state_path):
            instance._resid_centerer = joblib.load(state_path)
        return instance


# -- Nicht-Mut EN auf gene_expression-only (Basiszellen des Faktorials, gene_list via yaml) --------
# Die drevalpy-Basis `ElasticNet` nutzt laut yaml gene_expression + proteomics; für einen sauberen,
# intern konsistenten 2x2x2-Vergleich brauchen wir eine reine gene_expression-Variante mit gene_list.


class ElasticNetGeneExpr(_GeneListMixin, ElasticNetModel):
    """ElasticNet, nur gene_expression + fingerprints, Genliste via `gene_list`-hp (270 default)."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetGeneExpr"""
        return "ElasticNetGeneExpr"


class ElasticNetGeneExpr893(ElasticNetGeneExpr):
    """Wie ElasticNetGeneExpr, aber gene_list=landmark_plus_clinical_drivers (893) via yaml."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetGeneExpr893"""
        return "ElasticNetGeneExpr893"


# -- Resid-Varianten (kein Mut-Feature) -----------------------------------------------------------


class ElasticNetResid(_DrugResidMixin, ElasticNetGeneExpr):
    """ElasticNet + Drug-Mean-Residualisierung, gene_expression-only, 270 (gene_list via yaml)."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetResid"""
        return "ElasticNetResid"


class ElasticNetResid893(_DrugResidMixin, ElasticNetGeneExpr893):
    """ElasticNet + Drug-Mean-Residualisierung, gene_expression-only, 893."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetResid893"""
        return "ElasticNetResid893"


# -- Resid-Varianten MIT target-gematchtem Mutationsfeature ---------------------------------------


class ElasticNetTargetMutResid(_DrugResidMixin, ElasticNetTargetMut):
    """ElasticNet + Target-Mut-Feature + Drug-Mean-Residualisierung, 270."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetTargetMutResid"""
        return "ElasticNetTargetMutResid"


class ElasticNetTargetMutResid893(_DrugResidMixin, ElasticNetTargetMut893):
    """ElasticNet + Target-Mut-Feature + Drug-Mean-Residualisierung, 893."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetTargetMutResid893"""
        return "ElasticNetTargetMutResid893"
