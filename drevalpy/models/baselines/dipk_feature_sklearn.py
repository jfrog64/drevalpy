"""
RandomForest auf DIPKs Feature-Bausteinen (Gen-Autoencoder + gepooltes MolGNet) statt DIPKs DNN.

Auftrag Claude Science 2026-08-05 (Idee des Users): DIPKs DNN bringt an der eigentlichen
Within-Tissue-Interaktion keinen Mehrwert über RF+Mut (0,324 vs 0,322). Frage: Was, wenn man
DIPKs FEATURE-Bausteine behält — die AE-kodierte Genexpression und das MolGNet-Drug-Embedding —
aber den DNN durch einen RandomForest ersetzt, ± das target-gematchte Mut-Feature?
Erkenntniswert ist die FALSIFIKATION: bringen gelernte Embeddings dem RF überhaupt etwas, oder
ist RF auf rohen Features + Mut bereits das Optimum?

Faktorieller Ablations-Aufbau (je ±Mut, alles andere identisch zur RF-Baseline):
  Zelllinien-Feature            Drug-Feature          Klasse
  rohe gene_expr (893)          Fingerprints (128)    RandomForest / RandomForestTargetMut893  (Referenz, existiert)
  CTRPv2-AE (512)               Fingerprints (128)    RandomForestAE  / RandomForestAEMut
  rohe gene_expr (893)          MolGNet-pooled        RandomForestMolGNet / RandomForestMolGNetMut
  CTRPv2-AE (512)               MolGNet-pooled        RandomForestAEMolGNet / RandomForestAEMolGNetMut

--------------------------------------------------------------------------------------------
DESIGN-ENTSCHEIDUNGEN UND ABWEICHUNGEN VOM AUFTRAG (bewusst, hier dokumentiert)
--------------------------------------------------------------------------------------------
1. **MolGNet-Dimension: 768, nicht 43.** Der Auftrag nennt „~29 Atome × 43 Dims → 86 gepoolte
   Dims". Die Dateien `data/CTRPv2/DIPK_features/Drugs/MolGNet_<pubchem>.csv` sind TAB-separiert
   und haben tatsächlich die Form (n_Atome, **768**) — z.B. (16, 768), (32, 768), (36, 768).
   mean+max-Pooling ergibt also **1536** Drug-Features, nicht 86. Für den RF heißt das: die
   MolGNet-Arme haben ~2400 statt ~1000 Features und sind damit die teuersten Zellen des
   Faktorials (Split-Suche skaliert linear in der Featurezahl).
2. **AE-Trainingsmatrix: eindeutige Trainings-Zelllinien statt Response-Zeilen.** DIPK füttert dem
   AE die Genexpressionsmatrix ÜBER DIE RESPONSE-ZEILEN (`get_feature_matrix(identifiers=
   output.cell_line_ids)`), d.h. jede Zelllinie ~380-fach dupliziert (175k × 11.883 ≈ 8,3 GB).
   Da der AE unüberwacht auf ZELLLINIEN lernt, ist das nur eine Gewichtung nach Response-Anzahl
   (bei CTRPv2 nahezu uniform) plus sehr viele Gradientenschritte. Wir trainieren auf den ~815
   eindeutigen Trainingszelllinien und gleichen die Schrittzahl über `ae_batch_size` (64 statt
   1024) und mehr Epochen aus. Validierung fürs Early Stopping: 10 % der Trainingszelllinien
   (deterministisch, `ae_seed`) — sklearn-Modelle haben kein `output_earlystopping`, aus dem DIPK
   seine AE-Validierung zieht.
3. **AE-Cache über die Arme hinweg.** Der Encoder hängt nur von (Trainingszelllinien, Genliste,
   AE-Hyperparametern) ab, nicht vom Drug-Feature. Ohne Cache würde derselbe AE 4× trainiert
   (AE±Mut, AE+MolGNet±Mut). Mit Cache (Verzeichnis über `DREVAL_AE_CACHE` oder Hyperparameter
   `ae_cache_dir`) wird er je Fold EINMAL trainiert und wiederverwendet. Das spart nicht nur Zeit,
   es macht die Arme auch paarweise vergleichbar: identischer Encoder → Unterschiede zwischen
   AE und AE+MolGNet stammen garantiert nur vom Drug-Feature.
4. **Genraum des AE: `tcga_autoencoder_genes_available` (11.883) + log2(TPM+1)**, exakt wie
   DIPKLOG. Grund (siehe `DIPKLOG/dipklog.py`): CTRPv2-Genexpression ist LINEARE TPM; auf linearer
   Skala tragen die Top-100-Gene ~80 % der Varianz, ein MSE-AE rekonstruiert dann nur Housekeeper.
5. Der AE-Output läuft danach durch die normale `scale_gene_expression`-Kette der sklearn-Modelle
   (arcsinh + StandardScaler). Für den RF irrelevant (monoton je Feature), aber so bleibt der Pfad
   identisch zu allen anderen Baselines.

Nicht upstream-tauglich in dieser Form (Playground-Klon, DrEval-spezifische Defaults).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..DIPK.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)
from .bionic_features import _BNFMixin
from .sklearn_models import RandomForest, SklearnModel
from .target_mut_sklearn import _GeneListMixin, _TargetMutMixin

#: View-Name des gepoolten MolGNet-Embeddings (eigener Name, damit er nicht mit DIPKs
#: knoten-level `molgnet_features` verwechselt werden kann).
MOLGNET_POOLED_VIEW = "molgnet_pooled"

#: Genliste des Autoencoders — identisch zu DIPKLOG.
AE_GENE_LIST = "tcga_autoencoder_genes_available"


# ==================================================================================================
# Autoencoder-kodierte Genexpression
# ==================================================================================================


class _AEGeneExpressionMixin:
    """Mixin: ersetzt die rohe Genexpression durch die 512-dim Kodierung eines Fold-eigenen AE.

    Der AE (`GeneExpressionEncoder`, DIPKs Bausteine) wird je Fold FROM SCRATCH auf den
    Trainingszelllinien trainiert (kein Leck: Testzelllinien gehen nie ins AE-Training ein, sie
    werden nur mit dem fertigen Encoder transformiert — dasselbe Vorgehen wie in DIPK).
    """

    def build_model(self, hyperparameters: dict) -> None:
        """Basismodell bauen und die AE-Konfiguration aus den Hyperparametern ziehen.

        Zusätzliche (optionale) Hyperparameter:
          ae_log2 (True), epochs_autoencoder (1000), ae_batch_size (64), ae_patience (20),
          ae_lr (1e-4), ae_val_fraction (0.1), ae_seed (42), ae_cache_dir (None → env
          DREVAL_AE_CACHE, sonst kein Cache)

        :param hyperparameters: Hyperparameter des Modells
        """
        super().build_model(hyperparameters)
        hp = self.hyperparameters
        self._ae_log2 = bool(hp.get("ae_log2", True))
        self._ae_epochs = int(hp.get("epochs_autoencoder", 1000))
        self._ae_batch_size = int(hp.get("ae_batch_size", 64))
        self._ae_patience = int(hp.get("ae_patience", 20))
        self._ae_lr = float(hp.get("ae_lr", 1e-4))
        self._ae_val_fraction = float(hp.get("ae_val_fraction", 0.1))
        self._ae_seed = int(hp.get("ae_seed", 42))
        self._ae_cache_dir = hp.get("ae_cache_dir") or os.environ.get("DREVAL_AE_CACHE") or None
        self._encoder: GeneExpressionEncoder | None = None
        self._ae_input_dim: int | None = hp.get("gene_encoder_input_dim")

    # -- Feature-Laden ----------------------------------------------------------------------
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Genexpression auf dem AE-Genraum laden und (default) nach log2(TPM+1) transformieren.

        Die Genliste kommt über den Hyperparameter `gene_list` (Basisverhalten von `SklearnModel`);
        gesetzt wird sie in der hyperparameters.yaml auf `tcga_autoencoder_genes_available`.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset mit (ggf. log2-transformierter) Genexpression
        """
        cell_line_input = super().load_cell_line_features(data_path, dataset_name)
        if self._ae_log2:
            cell_line_input.apply(lambda x: np.log2(np.clip(x, 0.0, None) + 1.0), view="gene_expression")
        return cell_line_input

    # -- AE-Training / -Cache ---------------------------------------------------------------
    def _ae_config(self) -> dict:
        """:returns: die für den Encoder identitätsstiftende Konfiguration (geht in den Cache-Key)."""
        return {
            "gene_list": self.hyperparameters.get("gene_list", AE_GENE_LIST),
            "log2": self._ae_log2,
            "epochs": self._ae_epochs,
            "batch_size": self._ae_batch_size,
            "patience": self._ae_patience,
            "lr": self._ae_lr,
            "val_fraction": self._ae_val_fraction,
            "seed": self._ae_seed,
        }

    def _cache_path(self, train_ids: np.ndarray, input_dim: int, genes: list[str]) -> str | None:
        """Pfad der Cache-Datei für genau diese Trainingszelllinien + AE-Konfiguration + Gen-REIHENFOLGE.

        Die Reihenfolge gehört in den Schlüssel, seit der Upstream-Fix ``fadec10b`` (übernommen
        2026-09-04) die Feature-Spalten auf die Reihenfolge der Genliste umsortiert statt sie in
        CSV-Spaltenordnung zu lassen. Beide Ordnungen unterscheiden sich für JEDE hier benutzte
        Genliste. Ohne die Reihenfolge im Schlüssel würde ein vor dem Fix trainierter Encoder unter
        demselben Schlüssel geladen und mit vertauschten Genen gefüttert — kein Fehler, nur falsche
        Zahlen. Alte Cache-Dateien greifen deshalb nicht mehr; der Encoder wird einmalig neu
        trainiert (~5 min je Fold).

        :param train_ids: sortierte, eindeutige Trainingszelllinien
        :param input_dim: Anzahl Gene
        :param genes: Gen-Symbole in genau der Reihenfolge, in der sie in den Encoder gehen
        :returns: Dateipfad oder None, wenn kein Cache-Verzeichnis konfiguriert ist
        """
        if not self._ae_cache_dir:
            return None
        payload = json.dumps(
            {
                "cfg": self._ae_config(),
                "input_dim": int(input_dim),
                "cell_lines": sorted(map(str, train_ids)),
                "gene_order": hashlib.md5("\n".join(map(str, genes)).encode()).hexdigest(),
            },
            sort_keys=True,
        )
        digest = hashlib.md5(payload.encode()).hexdigest()[:16]
        return os.path.join(self._ae_cache_dir, f"gene_ae_{digest}.pt")

    def _fit_or_load_encoder(self, train_ids: np.ndarray, cell_line_input: FeatureDataset) -> GeneExpressionEncoder:
        """Encoder für diesen Fold beschaffen: aus dem Cache laden oder neu trainieren (und cachen).

        :param train_ids: eindeutige Trainingszelllinien dieses Folds
        :param cell_line_input: Zelllinien-Features (Genexpression, noch unkodiert)
        :returns: trainierter Encoder
        """
        matrix = cell_line_input.get_feature_matrix(view="gene_expression", identifiers=train_ids)
        self._ae_input_dim = int(matrix.shape[1])

        gene_order = [str(g) for g in cell_line_input.meta_info["gene_expression"]]
        path = self._cache_path(train_ids, self._ae_input_dim, gene_order)
        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            encoder = GeneExpressionEncoder(state["input_dim"])
            encoder.load_state_dict(state["encoder"])
            encoder.eval()
            print(
                f"[AE] Encoder aus Cache: {path} "
                f"({len(train_ids)} Trainingszelllinien, {state['input_dim']} Gene)"
            )
            return encoder

        # deterministischer Train/Val-Split der Zelllinien fürs Early Stopping des AE
        rng = np.random.default_rng(self._ae_seed)
        order = rng.permutation(len(matrix))
        n_val = max(1, int(round(self._ae_val_fraction * len(matrix))))
        val_matrix, train_matrix = matrix[order[:n_val]], matrix[order[n_val:]]
        print(
            f"[AE] trainiere Gen-Autoencoder von Grund auf: {len(train_matrix)} Zelllinien "
            f"(+{len(val_matrix)} val), {self._ae_input_dim} Gene, batch={self._ae_batch_size}, "
            f"max {self._ae_epochs} Epochen, patience={self._ae_patience}"
        )
        encoder = train_gene_expession_autoencoder(
            train_matrix,
            val_matrix,
            epochs_autoencoder=self._ae_epochs,
            lr=self._ae_lr,
            patience=self._ae_patience,
            batch_size=self._ae_batch_size,
        )
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(
                {
                    "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                    "input_dim": self._ae_input_dim,
                    "config": self._ae_config(),
                    "n_train_cell_lines": int(len(train_ids)),
                },
                path,
            )
            print(f"[AE] Encoder gecacht: {path}")
        return encoder

    def _encode(self, cell_line_input: FeatureDataset) -> FeatureDataset:
        """Kodiert den gene_expression-View IN PLACE mit dem trainierten Encoder.

        In-place ist sicher, weil `experiment.py` jedem Modellaufruf ein `cl_features.copy()`
        übergibt (Deep Copy, s. `FeatureDataset.copy`). Bereits kodierte Datensätze werden
        erkannt und übersprungen.

        :param cell_line_input: Zelllinien-Features
        :returns: dasselbe (nun kodierte) FeatureDataset
        """
        first = next(iter(cell_line_input.features.values()))["gene_expression"]
        if self._encoder is not None and len(first) == self._encoder.latent_dim:
            return cell_line_input  # schon kodiert (derselbe Datensatz wurde erneut übergeben)
        cell_line_input.apply(
            lambda x: encode_gene_expression(x, self._encoder),  # type: ignore[arg-type]
            view="gene_expression",
        )
        return cell_line_input

    # -- train / predict --------------------------------------------------------------------
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """AE auf den Trainingszelllinien fitten, Genexpression kodieren, dann Standard-Training.

        :param output: Trainingsdaten (Response)
        :param cell_line_input: Zelllinien-Features
        :param drug_input: Drug-Features
        :param output_earlystopping: nicht genutzt (sklearn-Modelle)
        :param model_checkpoint_dir: nicht genutzt
        """
        if len(output) > 0:
            train_ids = np.unique(np.asarray(output.cell_line_ids).astype(str))
            self._encoder = self._fit_or_load_encoder(train_ids, cell_line_input)
            self.hyperparameters["gene_encoder_input_dim"] = self._ae_input_dim
            cell_line_input = self._encode(cell_line_input)
        super().train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
        )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Genexpression mit dem Fold-Encoder kodieren, dann Standard-Vorhersage.

        :param cell_line_ids: Zelllinien-IDs
        :param drug_ids: Drug-IDs
        :param cell_line_input: Zelllinien-Features
        :param drug_input: Drug-Features
        :returns: Vorhersagen
        """
        if self._encoder is not None:
            cell_line_input = self._encode(cell_line_input)
        return super().predict(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

    # -- Persistenz -------------------------------------------------------------------------
    _AE_FILE = "gene_encoder.pt"

    def save(self, directory: str) -> None:
        """Basis-Artefakte + den Fold-Encoder speichern (ohne ihn wäre das Modell nicht ladbar).

        :param directory: Zielverzeichnis
        """
        super().save(directory)
        if self._encoder is not None:
            torch.save(
                {
                    "encoder": {k: v.cpu() for k, v in self._encoder.state_dict().items()},
                    "input_dim": self._ae_input_dim,
                    "config": self._ae_config(),
                },
                os.path.join(directory, self._AE_FILE),
            )

    @classmethod
    def load(cls, directory: str) -> SklearnModel:
        """Basis-`load` + Wiederherstellung des Gen-Encoders.

        :param directory: Verzeichnis mit den gespeicherten Artefakten
        :returns: wiederhergestellte Modellinstanz
        """
        instance = super().load(directory)
        path = os.path.join(directory, cls._AE_FILE)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            encoder = GeneExpressionEncoder(state["input_dim"])
            encoder.load_state_dict(state["encoder"])
            encoder.eval()
            instance._encoder = encoder
            instance._ae_input_dim = state["input_dim"]
        return instance


# ==================================================================================================
# Gepooltes MolGNet-Drug-Embedding
# ==================================================================================================


def pool_molgnet(nodes: np.ndarray, pooling: str = "mean_max") -> np.ndarray:
    """Aggregiert eine knoten-level MolGNet-Matrix (n_Atome × 768) zu einem festen Vektor.

    :param nodes: Knoten-Embeddings eines Moleküls
    :param pooling: "mean", "max" oder "mean_max" (Konkatenation, default)
    :returns: gepoolter Vektor (768 bzw. 1536 Dimensionen)
    :raises ValueError: bei unbekannter Pooling-Variante
    """
    if pooling == "mean":
        return nodes.mean(axis=0)
    if pooling == "max":
        return nodes.max(axis=0)
    if pooling == "mean_max":
        return np.concatenate((nodes.mean(axis=0), nodes.max(axis=0)))
    raise ValueError(f"Unbekanntes MolGNet-Pooling: {pooling!r}")


class _MolGNetPooledMixin:
    """Mixin: ersetzt die Fingerprints durch das über die Atome gepoolte MolGNet-Embedding.

    DIPK poolt knoten-level MolGNet über Attention im DNN; ein RF braucht einen festen Vektor,
    daher mean+max über die Knoten (die einzige nicht-triviale Design-Entscheidung des Auftrags).
    """

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Lädt alle `MolGNet_<pubchem>.csv` (TAB-separiert) und poolt sie über die Atome.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset mit dem View `molgnet_pooled`
        :raises FileNotFoundError: wenn das DIPK-Drug-Verzeichnis fehlt
        """
        pooling = self.hyperparameters.get("molgnet_pooling", "mean_max")
        drug_path = os.path.join(data_path, dataset_name, "DIPK_features", "Drugs")
        if not os.path.isdir(drug_path):
            raise FileNotFoundError(f"MolGNet-Features nicht gefunden: {drug_path}")

        features, n_dim = {}, None
        for file in sorted(os.listdir(drug_path)):
            if not (file.startswith("MolGNet_") and file.endswith(".csv")):
                continue
            drug = file[len("MolGNet_") : -len(".csv")]
            nodes = pd.read_csv(os.path.join(drug_path, file), index_col=0, sep="\t").to_numpy(dtype=np.float32)
            vector = pool_molgnet(nodes, pooling).astype(np.float32)
            n_dim = len(vector)
            features[drug] = {MOLGNET_POOLED_VIEW: vector}

        node_dim = (n_dim // 2) if pooling == "mean_max" else n_dim
        if pooling == "mean_max":
            names = [f"molgnet_mean_{i}" for i in range(node_dim)] + [f"molgnet_max_{i}" for i in range(node_dim)]
        else:
            names = [f"molgnet_{pooling}_{i}" for i in range(node_dim)]
        print(f"[MolGNet] {len(features)} Drugs, Pooling={pooling}, {n_dim} Features (Knoten-Dim {node_dim})")
        return FeatureDataset(features=features, meta_info={MOLGNET_POOLED_VIEW: np.array(names)})


# ==================================================================================================
# Konkrete Modellklassen — 6 Zellen des Faktorials (die 2 Referenzzellen existieren bereits)
# ==================================================================================================
#
# MRO-Hinweis: der AE-Mixin steht IMMER vor `_TargetMutMixin`. Grund: `_TargetMutMixin` erbt von
# `_GeneListMixin`, dessen `load_cell_line_features` den gene_list-Hyperparameter selbst behandelt.
# Stünde er vorn, käme die log2-Transformation des AE-Mixins nie zum Zug.


class RandomForest893(_GeneListMixin, RandomForest):
    """Referenzzelle des Faktorials: RF auf roher Genexpression (893) + Fingerprints, OHNE Mut.

    Die registrierte `RandomForest`-Basis nutzt laut yaml die 270er Liste, das Gegenstück MIT Mut
    (`RandomForestTargetMut893`) aber 893 Gene — die −Mut-Referenzzelle auf 893 Genen fehlte damit
    bisher komplett (in `results/` liegt nur `RandomForest` mit 270). Ohne sie wäre die −Mut-Spalte
    des Ablations-Faktorials mit dem Genraum konfundiert.
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForest893"""
        return "RandomForest893"


class RandomForestAE(_AEGeneExpressionMixin, RandomForest):
    """RF auf AE-kodierter Genexpression (512) + Fingerprints, ohne Mut-Feature."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAE"""
        return "RandomForestAE"


class RandomForestAEMut(_AEGeneExpressionMixin, _TargetMutMixin, RandomForest):
    """RF auf AE-kodierter Genexpression (512) + Fingerprints + target-gematchtem Mut-Block."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMut"""
        return "RandomForestAEMut"


class RandomForestMolGNet(_MolGNetPooledMixin, RandomForest):
    """RF auf roher Genexpression (893) + gepooltem MolGNet, ohne Mut-Feature."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestMolGNet"""
        return "RandomForestMolGNet"


class RandomForestMolGNetMut(_MolGNetPooledMixin, _TargetMutMixin, RandomForest):
    """RF auf roher Genexpression (893) + gepooltem MolGNet + target-gematchtem Mut-Block."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestMolGNetMut"""
        return "RandomForestMolGNetMut"


class RandomForestAEMolGNet(_AEGeneExpressionMixin, _MolGNetPooledMixin, RandomForest):
    """Kern der Idee: RF auf DIPKs beiden Feature-Bausteinen (AE 512 + MolGNet-pooled), ohne Mut."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNet"""
        return "RandomForestAEMolGNet"


class RandomForestAEMolGNetMut(_AEGeneExpressionMixin, _MolGNetPooledMixin, _TargetMutMixin, RandomForest):
    """Kern der Idee + Mut-Feature: RF auf AE 512 + MolGNet-pooled + target-gematchtem Mut-Block."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMut"""
        return "RandomForestAEMolGNetMut"


class RandomForestAEMolGNetMutKern8(RandomForestAEMolGNetMut):
    """Wie RandomForestAEMolGNetMut, nur mit der auf den Panel-Kern beschraenkten Statusmatrix.

    Aufgabe A1, Option (b) (User 2026-09-10): Das Mut-Feature soll nur noch aus Genen gespeist
    werden, die in ALLEN SECHS Sequenzier-Panels der Zelllinien-Quellen enthalten sind -- dem
    8-Gen-Kern AKT1, BRAF, ERBB2, KRAS, NRAS, PIK3CA, PTEN, TP53. Damit haengt das Feature nicht
    mehr davon ab, welches Panel eine Zelllinie zufaellig durchlaufen hat.

    Der einzige Unterschied zur Elternklasse ist der Pfad der Statusmatrix (Hyperparameter
    ``target_status_path`` in hyperparameters.yaml, gesetzt auf
    ``functional_status_matrix_kern8_20260910.csv``). Kein Code in _TargetMutMixin aendert sich:
    Gene, die in der CSV fehlen, fallen im Lookup automatisch auf Status ``missing`` -- das
    betrifft 23 der 48 kuratierten Substanzen (EGFR 9, JAK2 5, MET 3, ALK 2, FLT3 2, KIT 1,
    FGFR3 1). Von den Aggregaten bleibt PI3K_AKT identisch (alle drei Komponenten im Kern),
    RAS_MAPK wird OHNE NF1 neu gebildet und aendert sich dadurch bei 72 von 856 Zelllinien.
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutKern8"""
        return "RandomForestAEMolGNetMutKern8"


class RandomForestAEMolGNetMutBNF(_AEGeneExpressionMixin, _BNFMixin, _MolGNetPooledMixin, _TargetMutMixin, RandomForest):
    """Wie RandomForestAEMolGNetMut, zusaetzlich mit DIPKs DRITTEM Baustein BNF (BIONIC, 512).

    Aufgabe L1 (User 2026-09-11). Unser Ablations-Faktorial hat bisher nur ZWEI der drei
    DIPK-Bausteine geprueft (AE-Genexpression und MolGNet); der dritte -- der gemittelte
    BIONIC-Netzwerkvektor der hoechstexprimierten Gene -- fehlte komplett. Die bisherige Aussage
    "DIPKs Feature-Bausteine bringen dem RF nichts" stand damit auf zwei von drei Beinen.

    Diese Klasse nutzt die Auswahlregel, die drevalpy tatsaechlich implementiert
    (``DIPK/data_utils.py:load_bionic_features``): Rangliste ueber die UNGEFILTERTE
    Expressionsmatrix (CTRPv2: 42 209 Gene), ``gene_add_num=512``, erst danach Filter gegen
    ``gene_list_sel`` und die BIONIC-Tabelle. Gemessen bleiben davon im Median nur 120 gemittelte
    Vektoren uebrig. Das Gegenstueck nach Publikation ist ``RandomForestAEMolGNetMutBNForig``.

    MRO: ``_BNFMixin`` steht VOR ``_TargetMutMixin``, weil dessen Basis ``_GeneListMixin`` bei
    gesetztem ``gene_list`` ohne ``super()``-Aufruf zurueckkehrt -- dahinter wuerde BNF still
    ausfallen.
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutBNF"""
        return "RandomForestAEMolGNetMutBNF"


class RandomForestAEMolGNetMutBNForig(RandomForestAEMolGNetMutBNF):
    """Wie RandomForestAEMolGNetMutBNF, aber mit der BNF-Auswahlregel der PUBLIKATION.

    Unterschied allein in zwei Hyperparametern (``bnf_variant: orig``, ``bnf_gene_add_num: 256``):
    die Rangliste laeuft INNERHALB von ``gene_list_sel`` (CTRPv2: 5472 vorhandene Gene) und nimmt
    deren Top-256 -- so beschreiben es Li et al. (Brief Bioinform 25(3), bbae153, 2024) und so
    setzen es sowohl das Original-Repo user15632/DIPK (``DataConfig.py``: ``gene_add_num = 256``)
    als auch daisybios eigenes ``preprocess_drp_data`` um. Es bleiben im Median 215 statt 120
    gemittelte Vektoren.

    Der Arm existiert, damit die Frage "aendert drevalpys Abweichung das Ergebnis?" beantwortbar
    ist, bevor daraus ein Fehlerbericht wird (Aufgabe L2).
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutBNForig"""
        return "RandomForestAEMolGNetMutBNForig"
