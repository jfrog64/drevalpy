"""
BNF — DIPKs dritter Feature-Baustein (BIONIC-Netzwerk-Embeddings) fuer die sklearn-Arme.

Auftrag User 2026-09-11 („Bitte als Auftrag in Arbeitstabelle, dann implementieren und starten",
Arbeitstabelle Abschnitt L1).

--------------------------------------------------------------------------------------------------
WAS BNF IST
--------------------------------------------------------------------------------------------------
DIPK hat DREI Bausteine, unser Nachbau (`dipk_feature_sklearn.py`) hatte bisher nur zwei:

  GEF  AE-kodierte Genexpression (512)      -> `_AEGeneExpressionMixin`   uebernommen
  MolGNet Drug-Embedding (gepoolt, 1536)    -> `_MolGNetPooledMixin`      uebernommen
  BNF  gemittelte BIONIC-Vektoren (512)     -> DIESES MODUL               fehlte komplett

BNF entsteht so: `human_ppi_features.tsv` ist eine feste Tabelle Gen -> 512-dim Vektor (12 938 Gene),
gelernt von BIONIC (Forster et al., Nat Methods 2022) aus mehreren biologischen Netzwerken.
Die Tabelle ist fuer ALLE Zelllinien und ALLE Datensaetze identisch. Zelllinien-spezifisch wird sie
erst durch die AUSWAHL: pro Zelllinie werden die hoechstexprimierten Gene genommen und deren
BIONIC-Vektoren gemittelt. Das Merkmal sagt also „in welcher Netzwerk-Region ist diese Zelllinie
gerade aktiv".

Publikation (Li et al., Brief Bioinform 25(3), bbae153, 2024), woertlich:
  „the top 256 genes with the highest expression levels" -> „the average of these highly expressed
  gene representations".

--------------------------------------------------------------------------------------------------
DIE ZWEI VARIANTEN — UND WARUM BEIDE GERECHNET WERDEN
--------------------------------------------------------------------------------------------------
drevalpys `DIPK/data_utils.py:load_bionic_features` weicht an ZWEI Stellen von der Publikation ab:

  1. `gene_add_num` ist **512**, nicht 256 (Original-Repo `Task1/fold=0_model=0/DataConfig.py`
     und daisybios eigenes `preprocess_drp_data/utils/DIPK_features/PreprocessingA/PreprocessingA.py`
     sagen beide 256).
  2. Die Rangliste laeuft ueber die **ungefilterte** `gene_expression.csv` (CTRPv2: 42 209 Gene) und
     filtert ERST DANACH gegen `gene_list_sel`. Im Original ist die Expressionsmatrix von vornherein
     auf `gene_list_sel` (5757 Gene) beschraenkt, die Top-k werden also INNERHALB dieser Liste gezogen.

Folge (gemessen auf CTRPv2, 1019 Zelllinien):

  Variante   | Genpool der Rangliste | k   | tatsaechlich gemittelte Vektoren
  -----------|-----------------------|-----|---------------------------------
  "drevalpy" | alle 42 209 Gene      | 512 | Median **120** (71-200)
  "orig"     | gene_list_sel (5472)  | 256 | Median **215** (194-236)

Die Abweichungen kompensieren sich NICHT: drevalpy mittelt ueber weniger Gene, obwohl es die
doppelte Zahl anfordert. Beide Varianten werden gerechnet, damit die Frage „aendert die Abweichung
das Ergebnis?" beantwortbar ist — das ist die Grundlage fuer L2 (Issue an daisybio).

--------------------------------------------------------------------------------------------------
DESIGN-ENTSCHEIDUNGEN
--------------------------------------------------------------------------------------------------
1. **Eigenes Einlesen von `gene_expression.csv` statt Wiederverwendung des `gene_expression`-Views.**
   Zwingend: `_AEGeneExpressionMixin` reduziert den View auf die AE-Genliste, log2-t ihn und ersetzt
   ihn dann durch den 512-dim AE-Code. Fuer die BNF-Rangliste braucht es die VOLLE Matrix. Genau so
   macht es auch drevalpys `load_bionic_features` (liest die CSV direkt von der Platte).

2. **Kein Leck.** BNF wird je Zelllinie ALLEIN aus deren eigener Expression berechnet; nichts wird
   ueber Zelllinien hinweg gefittet (kein Scaler, keine PCA, kein Mittelwert ueber den Trainingssatz).
   Test-Zelllinien beeinflussen die Trainings-Features also nicht und umgekehrt. Identisch zu DIPK.

3. **Die log2-Frage stellt sich hier nicht.** Die Auswahl ist eine RANGLISTE, und Raenge sind unter
   jeder streng monotonen Transformation invariant. `ae_log2` des GEF-Zweigs hat auf BNF keinen
   Einfluss — der Vergleich ist damit nicht mit der Skalenfrage konfundiert.

4. **Keine Standardisierung.** Empfaenger ist ein RandomForest, und Baeume sind gegen monotone
   Transformationen einzelner Spalten invariant. (Fuer lineare Modelle waere sie noetig — dann
   `bnf_standardize: true`, der Scaler wird dann wie bei `_TargetMutMixin` mitgespeichert.)

5. **Modulweiter Zwischenspeicher.** Das Einlesen kostet ~40 s (83 MB TSV + 42 209-Spalten-CSV).
   Der Schluessel enthaelt alles, was das Ergebnis bestimmt; die Dateien selbst sind unveraenderlich.

6. **Lautes Scheitern.** Lehre aus `gp20260903_curveweight` (ein Haken blieb ab Fold 3 stumm) und aus
   der toleranten `_load_target_tables`: fehlende Dateien oder ein leerer Lookup wuerden hier einen
   NULLBLOCK trainieren, ohne dass der Lauf abbricht. Deshalb wirft `_compute_bnf` bei fehlenden
   Dateien, und `_bnf_block` zaehlt fehlende Zelllinien und schreibt eine WARN-Zeile, auf die das
   Laufskript greppt.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import FeatureDataset

from .sklearn_models import SklearnModel

#: Genau die Auswahlregeln, die im Original bzw. in drevalpy implementiert sind.
BNF_VARIANTS = ("drevalpy", "orig")

#: Vorgabe fuer `gene_add_num` je Variante — "orig" folgt der Publikation (256), "drevalpy" dem Code (512).
BNF_DEFAULT_K = {"drevalpy": 512, "orig": 256}

_BNF_CACHE: dict[tuple, tuple[dict[str, np.ndarray], int]] = {}

#: Zuletzt geladene (data_path, dataset_name) dieses Prozesses. Notwendig, weil
#: `experiment._load_features_cached` den Loader NICHT aufruft, wenn die Feature-Matrix schon im
#: `_FEATURE_CACHE` liegt (siehe `_ensure_bnf`) — die Instanz kennt den Datenpfad dann nicht.
_BNF_LAST_SOURCE: tuple[str, str] | None = None


def _compute_bnf(
    data_path: str, dataset_name: str, variant: str, gene_add_num: int
) -> tuple[dict[str, np.ndarray], int]:
    """Berechnet je Zelllinie den gemittelten BIONIC-Vektor.

    :param data_path: Pfad zu den Daten, z.B. "data/"
    :param dataset_name: Name des Datensatzes, z.B. "CTRPv2"
    :param variant: "drevalpy" (Top-k aus ALLEN Genen, dann Filter) oder "orig" (Top-k INNERHALB
        `gene_list_sel`) — siehe Modul-Docstring
    :param gene_add_num: Anzahl der hoechstexprimierten Gene, ueber die gemittelt wird
    :returns: (Abbildung Zelllinienname -> 512-dim Vektor, Dimension)
    :raises ValueError: bei unbekannter Variante
    :raises FileNotFoundError: wenn Expressionsmatrix, Genliste oder BIONIC-Tabelle fehlen
    """
    if variant not in BNF_VARIANTS:
        raise ValueError(f"Unbekannte BNF-Variante {variant!r}, erlaubt: {BNF_VARIANTS}")

    key = (os.path.abspath(data_path), dataset_name, variant, int(gene_add_num))
    if key in _BNF_CACHE:
        return _BNF_CACHE[key]

    base = os.path.join(data_path, dataset_name)
    expr_path = os.path.join(base, "gene_expression.csv")
    gene_list_path = os.path.join(base, "DIPK_features", "gene_list_sel.txt")
    ppi_path = os.path.join(base, "DIPK_features", "human_ppi_features.tsv")
    for path in (expr_path, gene_list_path, ppi_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[BNF] Datei fehlt: {path}")

    expr = pd.read_csv(expr_path)
    expr = expr.set_index("cell_line_name")
    if "cellosaurus_id" in expr.columns:
        expr = expr.drop("cellosaurus_id", axis=1)

    # encoding="gbk" ist KEIN Tippfehler: so liest es das Original-DIPK (BIONIC_dict.py) und so
    # liest es drevalpy (DIPK/data_utils.py:36). Uebernommen, damit die Genliste bitgleich ist.
    with open(gene_list_path, encoding="gbk") as handle:
        gene_list = {line.strip() for line in handle if line.strip()}

    ppi = pd.read_csv(ppi_path, index_col=0, sep="\t")
    # Nur Gene, die BEIDES haben: in gene_list_sel und mit BIONIC-Zeile. Genau `bionic_gene_dict`
    # aus drevalpys data_utils.py:44.
    bionic = {gene: ppi.loc[gene].to_numpy(dtype=np.float32) for gene in gene_list if gene in ppi.index}
    if not bionic:
        raise FileNotFoundError(f"[BNF] Schnitt aus {gene_list_path} und {ppi_path} ist LEER")
    dim = len(next(iter(bionic.values())))

    if variant == "orig":
        # Original: die Expressionsmatrix ist von vornherein auf gene_list_sel beschraenkt, die
        # Rangliste laeuft also INNERHALB dieser Liste (DIPK Task1/DataPreprocess/RMA/RMA_dict.py
        # + Task1/fold=0_model=0/Data.py).
        cols = [c for c in expr.columns if c in gene_list]
        expr = expr[cols]

    genes = np.asarray(expr.columns, dtype=object)
    values = expr.to_numpy(dtype=np.float32)
    k = min(int(gene_add_num), values.shape[1])

    features: dict[str, np.ndarray] = {}
    hits = []
    for row, cell_line in enumerate(expr.index.astype(str)):
        # Absteigend sortieren, Gleichstaende in Spaltenreihenfolge — bitgleich zu drevalpys
        # `sorted(expressions.items(), key=lambda x: -x[1])` (Pythons sort ist stabil).
        # Zeilenweise, damit nicht 1019 x 42 209 Indizes gleichzeitig im Speicher liegen.
        order = np.argsort(-values[row], kind="stable")[:k]
        selected = [bionic[g] for g in genes[order] if g in bionic]
        hits.append(len(selected))
        # Wie im Original: Mittelwert ueber die GEFUNDENEN (Division durch k_gefunden, nicht durch k).
        features[cell_line] = (
            np.mean(selected, axis=0).astype(np.float32) if selected else np.zeros(dim, dtype=np.float32)
        )

    hits_arr = np.asarray(hits)
    print(
        f"[BNF] Variante {variant!r}: {len(features)} Zelllinien, Genpool {values.shape[1]}, "
        f"k={k}, BIONIC-faehige Gene {len(bionic)}, gemittelte Vektoren je Zelllinie "
        f"Median {int(np.median(hits_arr))} (Spanne {hits_arr.min()}-{hits_arr.max()}), Dim {dim}"
    )
    if hits_arr.min() == 0:
        print(f"[BNF] WARN: {int((hits_arr == 0).sum())} Zelllinien ohne EIN einziges BIONIC-Gen -> Nullvektor")

    _BNF_CACHE[key] = (features, dim)
    return features, dim


class _BNFMixin:
    """Mixin: haengt DIPKs BNF-Block (gemittelte BIONIC-Vektoren, 512) rechts an die Feature-Matrix.

    MRO: dieser Mixin muss VOR `_TargetMutMixin` stehen. Grund: `_TargetMutMixin` erbt von
    `_GeneListMixin`, dessen `load_cell_line_features` bei gesetztem `gene_list` direkt
    zurueckkehrt, OHNE `super()` zu rufen — stuende `_BNFMixin` dahinter, wuerde sein
    `load_cell_line_features` nie ausgefuehrt und der BNF-Block bliebe still leer.
    """

    #: Klassenvorgaben, damit `load_cell_line_features` auch ohne vorheriges `build_model` traegt.
    _bnf_variant: str = "drevalpy"
    _bnf_gene_add_num: int = BNF_DEFAULT_K["drevalpy"]

    def build_model(self, hyperparameters: dict) -> None:
        """Basismodell bauen und die BNF-Konfiguration aus den Hyperparametern ziehen.

        Zusaetzliche (optionale) Hyperparameter:
          bnf_variant       "drevalpy" (default) | "orig"
          bnf_gene_add_num  Anzahl hoechstexprimierter Gene (Default je Variante: 512 bzw. 256)
          bnf_standardize   False (default) — fuer RF unnoetig, fuer lineare Modelle noetig

        :param hyperparameters: Hyperparameter des Modells
        """
        super().build_model(hyperparameters)
        hp = self.hyperparameters
        self._bnf_variant = str(hp.get("bnf_variant", type(self)._bnf_variant))
        self._bnf_gene_add_num = int(hp.get("bnf_gene_add_num", BNF_DEFAULT_K[self._bnf_variant]))
        self._bnf_standardize = bool(hp.get("bnf_standardize", False))
        self._bnf_scaler: StandardScaler | None = None
        self._bnf: dict[str, np.ndarray] = {}
        self._bnf_dim = 0

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Basis-Features laden und nebenher die BNF-Tabelle dieses Datensatzes berechnen.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: unveraenderte Basis-Features (BNF haengt an `self`, nicht am FeatureDataset —
            es ist ein ZWEITER Zelllinien-View, und `SklearnModel.train` verarbeitet nur
            `cell_line_views[0]`)
        """
        global _BNF_LAST_SOURCE
        cell_line_input = super().load_cell_line_features(data_path, dataset_name)
        _BNF_LAST_SOURCE = (data_path, dataset_name)
        self._bnf, self._bnf_dim = _compute_bnf(
            data_path=data_path,
            dataset_name=dataset_name,
            variant=self._bnf_variant,
            gene_add_num=self._bnf_gene_add_num,
        )
        self.bnf_feature_names = [f"bnf_{self._bnf_variant}_{i}" for i in range(self._bnf_dim)]
        return cell_line_input

    def _ensure_bnf(self) -> None:
        """Stellt sicher, dass `self._bnf` gefuellt ist, und holt die Tabelle notfalls selbst.

        Warum das noetig ist (Fehlschlag 2026-09-11, Lauf gp20260911_bnf, Falte 3):
        `experiment._load_features_cached` haelt die geladene Feature-Matrix in einem prozessweiten
        `_FEATURE_CACHE`, dessen Schluessel die Hyperparameter enthaelt. Bei einem TREFFER wird
        `load_cell_line_features` **gar nicht aufgerufen** — die Zeile "Loading cell line features"
        steht trotzdem im Log. Da `train_and_predict` je Falte eine FRISCHE Modellinstanz baut
        (`build_model`), blieb deren `self._bnf` dann leer.
        Der Treffer kommt erst ab Falte 3, weil der AE-Mixin in Falte 1 `gene_encoder_input_dim` in
        die Hyperparameter schreibt und damit den Cache-Schluessel aendert: Falte 1 (ohne) und
        Falte 2 (mit) sind Fehltreffer, ab Falte 3 trifft der Schluessel von Falte 2.

        `_TargetMutMixin` hat das Problem nicht, weil er seine Tabellen in `build_model` laedt, und
        `_AEGeneExpressionMixin` nicht, weil er seinen Encoder in `train` holt. BNF braucht aber
        `data_path`/`dataset_name`, die `build_model` nicht bekommt — daher der Umweg ueber die
        zuletzt geladene Quelle. Der Nachbau ist gratis: `_compute_bnf` trifft seinen eigenen Cache.

        :raises RuntimeError: wenn in diesem Prozess noch nie Zelllinien-Features geladen wurden
            (waere ein stiller Nullblock) oder der Mixin in der MRO hinter `_TargetMutMixin` steht
        """
        if self._bnf:
            return
        if _BNF_LAST_SOURCE is None:
            raise RuntimeError(
                "[BNF] Tabelle ist leer und es wurde in diesem Prozess nie eine Zelllinien-Quelle "
                "geladen. Pruefen: steht _BNFMixin in der MRO VOR _TargetMutMixin? "
                "(_GeneListMixin.load_cell_line_features kehrt bei gesetztem gene_list ohne "
                "super()-Aufruf zurueck.)"
            )
        data_path, dataset_name = _BNF_LAST_SOURCE
        self._bnf, self._bnf_dim = _compute_bnf(
            data_path=data_path,
            dataset_name=dataset_name,
            variant=self._bnf_variant,
            gene_add_num=self._bnf_gene_add_num,
        )
        self.bnf_feature_names = [f"bnf_{self._bnf_variant}_{i}" for i in range(self._bnf_dim)]

    def _bnf_block(self, cell_line_ids: np.ndarray) -> np.ndarray:
        """Baut die (n_Zeilen x 512)-Matrix fuer die gegebenen Zelllinien.

        :param cell_line_ids: Zelllinien je Response-Zeile
        :returns: BNF-Block
        :raises RuntimeError: wenn die BNF-Tabelle weder gesetzt noch nachladbar ist
        """
        self._ensure_bnf()
        ids = np.asarray(cell_line_ids).astype(str)
        zero = np.zeros(self._bnf_dim, dtype=np.float32)
        missing = 0
        block = np.empty((len(ids), self._bnf_dim), dtype=np.float32)
        for i, cell_line in enumerate(ids):
            vector = self._bnf.get(cell_line)
            if vector is None:
                vector, missing = zero, missing + 1
            block[i] = vector
        if missing:
            print(f"[BNF] WARN: {missing}/{len(ids)} Zeilen ohne BNF-Eintrag -> Nullvektor")
        return block

    def get_concatenated_features(
        self,
        cell_line_view,
        drug_view,
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> np.ndarray:
        """Basis-Feature-Matrix + BNF-Block (rechts angehaengt).

        :param cell_line_view: Zelllinien-View der Basis
        :param drug_view: Drug-View der Basis
        :param cell_line_ids_output: Zelllinien je Response-Zeile
        :param drug_ids_output: Drugs je Response-Zeile
        :param cell_line_input: Zelllinien-Features
        :param drug_input: Drug-Features
        :returns: Feature-Matrix mit angehaengtem BNF-Block
        """
        base = super().get_concatenated_features(
            cell_line_view=cell_line_view,
            drug_view=drug_view,
            cell_line_ids_output=cell_line_ids_output,
            drug_ids_output=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        block = self._bnf_block(cell_line_ids_output)
        if self._bnf_scaler is not None:
            block = self._bnf_scaler.transform(block)
        return np.concatenate((base, block.astype(base.dtype)), axis=1)

    def train(
        self,
        output,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping=None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Optionalen BNF-Standardisierer auf den Trainingszeilen fitten, dann Standard-Training.

        :param output: Trainingsdaten (Response)
        :param cell_line_input: Zelllinien-Features
        :param drug_input: Drug-Features
        :param output_earlystopping: durchgereicht
        :param model_checkpoint_dir: durchgereicht
        """
        if self._bnf_standardize and len(output) > 0:
            self._bnf_scaler = StandardScaler().fit(self._bnf_block(output.cell_line_ids))
        super().train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
        )

    # -- Persistenz -----------------------------------------------------------------------------
    # Der Basis-`load` ruft `build_model`, das `self._bnf` auf {} zuruecksetzt; ohne das Folgende
    # haette ein wiederhergestelltes Modell einen leeren Lookup und `_bnf_block` wuerde werfen.
    _BNF_STATE_FILE = "bnf_state.pkl"

    def save(self, directory: str) -> None:
        """Basis-Artefakte + die BNF-Tabelle dieses Folds speichern.

        :param directory: Zielverzeichnis
        """
        super().save(directory)
        joblib.dump(
            {
                "bnf": self._bnf,
                "dim": self._bnf_dim,
                "variant": self._bnf_variant,
                "gene_add_num": self._bnf_gene_add_num,
                "standardize": self._bnf_standardize,
                "scaler": self._bnf_scaler,
            },
            os.path.join(directory, self._BNF_STATE_FILE),
        )

    @classmethod
    def load(cls, directory: str) -> SklearnModel:
        """Basis-`load` + Wiederherstellung der BNF-Tabelle.

        :param directory: Verzeichnis mit den gespeicherten Artefakten
        :returns: wiederhergestellte Modellinstanz
        """
        instance = super().load(directory)
        path = os.path.join(directory, cls._BNF_STATE_FILE)
        if os.path.exists(path):
            state = joblib.load(path)
            instance._bnf = state["bnf"]
            instance._bnf_dim = state["dim"]
            instance._bnf_variant = state["variant"]
            instance._bnf_gene_add_num = state["gene_add_num"]
            instance._bnf_standardize = state["standardize"]
            instance._bnf_scaler = state["scaler"]
            instance.bnf_feature_names = [f"bnf_{state['variant']}_{i}" for i in range(state["dim"])]
        return instance
