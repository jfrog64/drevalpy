"""
Gepoolte sklearn-Modelle mit DRUG-TARGET-gematchtem funktionellem Mutationsstatus-Feature.

Kernidee (DrEval, 2026-07-17): Der Mutations-Effekt ist eine Drug×Gen-INTERAKTION
(eine BRAF-Mutation zählt nur für BRAF-Inhibitoren). Im gepoolten Modell mit roher
Mutationsmatrix müsste das Modell diese Interaktion selbst finden — bei RF verdünnt,
bei linearem ElasticNet unmöglich. Lösung: die Interaktion wird ins Feature
VORGERECHNET (Status GENAU des Target-Gens des jeweiligen Drugs), sodass sie zum
Haupteffekt wird, den auch EN direkt nutzt.

Das Feature ist ein (Drug × Zelllinie)-Feature — passt nicht in drevalpys
cell_line_views (pro Zelllinie) oder drug_views (pro Drug). Deshalb wird es in
`get_concatenated_features` erzeugt, wo pro Response-Zeile sowohl `cell_line_ids`
als auch `drug_ids` vorliegen, und an die Standard-Feature-Matrix angehängt.

Datenquellen (aus dem Mutationsstatus-Test, Claude Science 2026-07-17):
  - curated_drug_target_variant.csv : pubchem_id, biomarker_gene, sensitive_status
  - functional_status_matrix.csv    : sample(=cell_line_name), gene, status
                                      {activating|LoF|other|wildtype}
IDs passen direkt: drevalpy drug_ids = pubchem_id, cell_line_ids = cell_line_name.

Feature-Block je Zeile (default onehot + aligned), alles 0 für Drugs ohne kuratiertes
Target (→ sicheres Superset, stört die übrigen 522 Drugs nicht):
  targetstat_activating, _LoF, _other, _wildtype, _missing   (One-hot Status des Target-Gens)
  target_sensitizing   = 1, wenn Status == sensitive_status des Drugs (aus curated).
    Diese *ausgerichtete* Spalte ist der eigentliche Haupteffekt: für Onkogen-Inhibitoren
    ist "activating" sensitiv, für MDM2/TP53 ist "wildtype" sensitiv — beide zeigen hier 1,
    also EIN konsistent gerichtetes Signal, das auch das lineare EN direkt nutzen kann.

Nicht upstream-tauglich (hartkodierte Default-Pfade auf DrEval) — Playground-Klon.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..utils import load_and_select_gene_features
from .sklearn_models import ElasticNetModel, RandomForest, SklearnModel

# Default-Pfade der kuratierten Tabellen (Claude-Science-Deliverables)
_DEFAULT_CURATED = "/mnt/data/genomics/CDR/DrEval/analysis_science/curated_drug_target_variant.csv"
_DEFAULT_STATUS = "/mnt/data/genomics/CDR/DrEval/analysis_science/functional_status_matrix.csv"


class _TargetMutMixin:
    """Mixin, das die Feature-Matrix um das target-gematchte Mutationsstatus-Feature erweitert."""

    def build_model(self, hyperparameters: dict):
        """Baut das Basismodell und lädt die Target/Status-Tabellen.

        Zusätzliche (optionale) Hyperparameter:
          target_curated_path, target_status_path : Pfade zu den CSVs
          target_missing   : "missing" (default) | "wildtype" | (Kategorie für nicht-abgedeckte Zelllinien)
          target_include_aligned : bool (default True) — die ausgerichtete target_sensitizing-Spalte anhängen
        """
        super().build_model(hyperparameters)
        hp = self.hyperparameters
        curated_path = hp.get("target_curated_path", _DEFAULT_CURATED)
        status_path = hp.get("target_status_path", _DEFAULT_STATUS)
        self._target_missing = hp.get("target_missing", "missing")
        self._target_include_aligned = bool(hp.get("target_include_aligned", True))
        # Standardisieren des Target-Blocks: nötig für lineare Modelle (ElasticNet), sonst
        # schrumpft die L1-Penalty die dünn besetzten 0/1-Spalten auf 0. Für RF harmlos
        # (Bäume sind gegen affine Transformationen invariant). Default an.
        self._target_standardize = bool(hp.get("target_standardize", True))
        self._target_scaler = None

        # feste Spaltenreihenfolge (train == predict!)
        self._status_cats = ["activating", "LoF", "other", "wildtype", "missing"]
        self.target_feature_names = [f"targetstat_{c}" for c in self._status_cats]
        if self._target_include_aligned:
            self.target_feature_names = self.target_feature_names + ["target_sensitizing"]

        self._load_target_tables(curated_path, status_path)

    def _load_target_tables(self, curated_path: str, status_path: str) -> None:
        """Baut die Lookups (drug->gene, drug->sens, (cell_line,gene)->status) aus den CSVs.

        Tolerant bei fehlenden Dateien: setzt leere Lookups + Warnung statt zu crashen. So kann ein
        via `load()` wiederhergestelltes Modell auch ohne die CSVs auskommen (der State wird dort ohnehin
        aus `target_mut_state.pkl` überschrieben); der Feature-Block ist ein sicheres Superset (alles 0).
        """
        if not (os.path.exists(curated_path) and os.path.exists(status_path)):
            print(
                f"[TargetMut] WARN: kuratierte/Status-Tabelle nicht gefunden "
                f"({curated_path!r}, {status_path!r}) -> Target-Feature bleibt 0, "
                f"sofern nicht via load() wiederhergestellt."
            )
            self._drug2gene, self._drug2sens, self._status = {}, {}, {}
            return

        curated = pd.read_csv(curated_path, dtype={"pubchem_id": str})
        curated = curated.dropna(subset=["biomarker_gene", "pubchem_id"])
        self._drug2gene = dict(zip(curated["pubchem_id"], curated["biomarker_gene"]))
        self._drug2sens = dict(zip(curated["pubchem_id"], curated["sensitive_status"]))

        status = pd.read_csv(status_path)
        # Lookup (cell_line_name, gene) -> status
        self._status = {
            (s, g): st for s, g, st in zip(status["sample"], status["gene"], status["status"])
        }

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Wie Basis, aber optional mit frei wählbarer Genexpr-Liste über den Hyperparameter `gene_list`.

        Ohne `gene_list`: Basisverhalten (fest verdrahtete `landmark_genes_reduced`/270 in utils).
        Mit `gene_list` (z.B. `"landmark_plus_clinical_drivers"`/893): diese Liste wird geladen. Da der Wert
        in `hyperparameters.json` mitgespeichert wird, nutzt ein via `load()` wiederhergestelltes Modell
        automatisch dieselbe Auswahl bei train UND predict — sonst würde ein auf 893 trainiertes Modell beim
        Standard-predict 270 Features bekommen (Dimensions-Mismatch). Greift nur für den gene_expression-View.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset mit den (ggf. per gene_list reduzierten) Zelllinien-Features
        """
        gene_list = self.hyperparameters.get("gene_list", None)
        if gene_list and self.cell_line_views == ["gene_expression"]:
            return load_and_select_gene_features(
                feature_type="gene_expression", gene_list=gene_list,
                data_path=data_path, dataset_name=dataset_name,
            )
        return super().load_cell_line_features(data_path, dataset_name)

    def _target_feature_matrix(self, cell_line_ids: np.ndarray, drug_ids: np.ndarray) -> np.ndarray:
        """Erzeugt die (n_rows × n_target_features)-Matrix für die gegebenen (Zelllinie, Drug)-Paare."""
        n = len(drug_ids)
        df = pd.DataFrame(
            {"cl": np.asarray(cell_line_ids).astype(str), "drug": np.asarray(drug_ids).astype(str)}
        )
        df["gene"] = df["drug"].map(self._drug2gene)  # NaN für Drugs ohne kuratiertes Target
        df["sens"] = df["drug"].map(self._drug2sens)
        has_target = df["gene"].notna()
        # Status je (Zelllinie, Target-Gen); None wenn Gen NaN oder Zelllinie nicht abgedeckt
        df["status"] = [self._status.get((c, g)) if isinstance(g, str) else None
                        for c, g in zip(df["cl"], df["gene"])]
        # abgedeckt=aber-kein-Eintrag bzw. nicht in DepMap -> definierte Kategorie (default "missing")
        df.loc[has_target & df["status"].isna(), "status"] = self._target_missing

        m = np.zeros((n, len(self.target_feature_names)), dtype=np.float32)
        status_arr = df["status"].to_numpy()
        for j, c in enumerate(self._status_cats):
            m[:, j] = (status_arr == c)
        if self._target_include_aligned:
            aligned = has_target.to_numpy() & (status_arr == df["sens"].to_numpy())
            m[:, len(self._status_cats)] = aligned
        return m

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Fittet den Target-Standardisierer auf den Trainingszeilen, dann Standard-Training."""
        if self._target_standardize and len(output) > 0:
            tm_train = self._target_feature_matrix(output.cell_line_ids, output.drug_ids)
            self._target_scaler = StandardScaler().fit(tm_train)
        super().train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
        )

    def get_concatenated_features(
        self,
        cell_line_view,
        drug_view,
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> np.ndarray:
        """Basis-Feature-Matrix + target-gematchter Mutationsstatus-Block (rechts angehängt)."""
        base = super().get_concatenated_features(
            cell_line_view=cell_line_view,
            drug_view=drug_view,
            cell_line_ids_output=cell_line_ids_output,
            drug_ids_output=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        tm = self._target_feature_matrix(cell_line_ids_output, drug_ids_output)
        if self._target_scaler is not None:
            tm = self._target_scaler.transform(tm)
        return np.concatenate((base, tm.astype(base.dtype)), axis=1)

    # -- Persistenz -----------------------------------------------------------------
    # Der Basis-`save` legt model.pkl / hyperparameters.json / scaler.pkl (gene_expression) ab; sein
    # `load` ruft `build_model` (baut die Lookups aus den CSVs neu) UND setzt dabei `_target_scaler=None`.
    # Ohne das Folgende ginge also der gefittete Target-StandardScaler verloren -> ein geladenes Modell
    # bekäme den Target-Block UNstandardisiert, während die Koeffizienten auf standardisierten Input
    # trainiert wurden = falsche Vorhersagen (v.a. EN). Wir persistieren daher den kompletten Target-State
    # (Scaler + Lookups + Config) -> geladenes Modell ist selbst-enthaltend und CSV-drift-fest.
    _TARGET_STATE_FILE = "target_mut_state.pkl"

    def save(self, directory: str) -> None:
        """Basis-Artefakte speichern (model/hyperparameters/gene-expr-scaler) + Target-State dazu.

        :param directory: Zielverzeichnis
        """
        super().save(directory)  # legt model.pkl / hyperparameters.json / scaler.pkl an (raise wenn untrainiert)
        joblib.dump(
            {
                "drug2gene": self._drug2gene,
                "drug2sens": self._drug2sens,
                "status": self._status,
                "status_cats": self._status_cats,
                "target_feature_names": self.target_feature_names,
                "target_missing": self._target_missing,
                "target_include_aligned": self._target_include_aligned,
                "target_standardize": self._target_standardize,
                "target_scaler": self._target_scaler,
            },
            os.path.join(directory, self._TARGET_STATE_FILE),
        )

    @classmethod
    def load(cls, directory: str) -> SklearnModel:
        """Basis-`load` + Wiederherstellung des Target-States (überschreibt die CSV-Rekonstruktion).

        :param directory: Verzeichnis mit den gespeicherten Modell-Artefakten
        :returns: wiederhergestellte Modellinstanz
        """
        instance = super().load(directory)  # build_model (CSV-tolerant) + model.pkl + gene-expr-scaler
        state_path = os.path.join(directory, cls._TARGET_STATE_FILE)
        if os.path.exists(state_path):
            state = joblib.load(state_path)
            instance._drug2gene = state["drug2gene"]
            instance._drug2sens = state["drug2sens"]
            instance._status = state["status"]
            instance._status_cats = state["status_cats"]
            instance.target_feature_names = state["target_feature_names"]
            instance._target_missing = state["target_missing"]
            instance._target_include_aligned = state["target_include_aligned"]
            instance._target_standardize = state["target_standardize"]
            instance._target_scaler = state["target_scaler"]
        return instance


class RandomForestTargetMut(_TargetMutMixin, RandomForest):
    """Gepooltes RandomForest + target-gematchtes funktionelles Mutationsstatus-Feature."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestTargetMut"""
        return "RandomForestTargetMut"


class ElasticNetTargetMut(_TargetMutMixin, ElasticNetModel):
    """Gepooltes ElasticNet + target-gematchtes funktionelles Mutationsstatus-Feature."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: ElasticNetTargetMut"""
        return "ElasticNetTargetMut"
