"""Rausch-bewusstes Training: Trainingszeilen nach der Qualität ihrer Dosis-Wirkungs-Kurve wichten.

Hintergrund (CTRPv2, Analyse 2026-09-03): ``curvecurator`` liefert je Kurve mit ``pEC50Error`` einen
Fit-Fehler des pEC50 in log10-Einheiten. Bei **18,5 %** der (Zelllinie, Drug)-Paare liegt dieser
Fehler bei ≥ 1,0 — mehr als die gesamte pEC50-Streuung des Datensatzes (sd 1,05). Der IC50 dieser
Zeilen ist nicht identifiziert; das Modell lernt sie bisher trotzdem mit vollem Gewicht.

Das Rauschen ist stark substanzgebunden: der Anteil unbestimmbarer Kurven je Drug korreliert mit
Spearman 0,87 damit, ob der IC50 überhaupt im getesteten Konzentrationsbereich liegt. Die Modellgüte
fällt nicht gleitend, sondern bricht erst jenseits von ~50 % unbestimmbarer Kurven je Drug weg
(Median within-drug ρ 0,42 → 0,29 → 0,00). Deshalb zwei Varianten, die sich hier nur im
Hyperparameter ``curve_weight_mode`` unterscheiden:

``inverse_variance``
    Weiche Gewichtung ``w = 1 / (tau² + SE²)`` in Response-Einheiten — schlecht bestimmte Kurven
    zählen weniger, bleiben aber drin. ``tau`` deckelt das Gewicht perfekter Kurven.
``trim``
    Harter Schnitt: Zeilen jenseits der Unbestimmbarkeitsschwelle bekommen Gewicht 0 und fallen
    aus dem Fit. Erwarteter Zielkonflikt — die Information „wirkt hier gar nicht" geht mit verloren,
    was globalen Pearson/RMSE kosten kann, während die Rangfolge innerhalb eines Drugs gewinnen soll.

Die Fehlerspalte steht in pEC50-Einheiten (log10), die Response in LN_IC50. Umgerechnet wird mit
``curve_weight_slope`` = 2,1821 — der global gefittete Betrag von ``d LN_IC50 / d pEC50``
(LN_IC50 = −2,1821·pEC50 + 13,4012, r = −0,9695).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .dipk_feature_sklearn import RandomForestAEMolGNetMut

# Spalten der Datensatz-CSV, die die Kurvenqualität tragen.
_QUALITY_FILE_COLS = ["cell_line_name", "pubchem_id", "pEC50Error"]

#: Fehlertabellen je (Datenpfad, Datensatz) — **modulweit, nicht je Modellinstanz**.
#:
#: Warum nicht auf ``self``: ``experiment._load_features_cached`` memoisiert
#: ``load_cell_line_features`` über ``_FEATURE_CACHE``. Ab dem dritten Fold ist der Cache-Key
#: stabil, die Lade-Methode wird also nicht mehr aufgerufen — während je Fold eine frische
#: Modellinstanz gebaut wird. Eine Instanzvariable wäre dann leer, und das Training liefe
#: unbemerkt ungewichtet weiter (genau der Fehler im Lauf ``gp20260903_curveweight``, Folds 3–5).
#: Die Tabelle hängt ohnehin am Datensatz und nicht am Modell, also gehört sie hierher.
_ERROR_TABLES: dict[tuple[str, str], dict[tuple[str, str], float]] = {}


class _CurveQualityWeightMixin:
    """Hängt ein ``sample_weight`` aus ``pEC50Error`` in den ``fit``-Aufruf von ``SklearnModel`` ein.

    Erwartet, dass die Datensatz-CSV ``<data_path>/<dataset>/<dataset>.csv`` die Spalte
    ``pEC50Error`` führt (CTRPv2 und GDSC1 tun das). Fehlt sie oder die Datei, **bricht das
    Training ab**. Ein Rückfall auf ungewichtetes Training wäre hier kein tolerierbarer Notnagel,
    sondern eine stille Verfälschung: das Modell hieße weiter ``…CW``/``…Trim``, träfe aber
    dieselben Vorhersagen wie das ungewichtete Basismodell.
    """

    def build_model(self, hyperparameters: dict) -> None:
        """Basismodell bauen und die Gewichtungsparameter übernehmen.

        :param hyperparameters: Hyperparameter des Modells
        :raises ValueError: bei unbekanntem ``curve_weight_mode``
        """
        super().build_model(hyperparameters)
        self._cq_mode = str(hyperparameters.get("curve_weight_mode", "inverse_variance"))
        if self._cq_mode not in ("inverse_variance", "trim"):
            raise ValueError(f"curve_weight_mode muss 'inverse_variance' oder 'trim' sein, nicht {self._cq_mode!r}")
        self._cq_slope = float(hyperparameters.get("curve_weight_slope", 2.1821))
        self._cq_tau = float(hyperparameters.get("curve_weight_tau", 0.2))
        self._cq_bad = float(hyperparameters.get("curve_weight_bad_pec50_error", 1.0))

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Zelllinien-Features laden und dabei die Kurvenfehler des Datensatzes einlesen.

        Der Haken sitzt hier, weil ``train`` den Datenpfad nicht kennt, diese Methode aber schon.
        Sie läuft allerdings **nicht** je Fold — der Aufrufer memoisiert sie (siehe
        ``_ERROR_TABLES``) —, deshalb landet das Ergebnis in einem modulweiten Cache.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset des Basismodells, unverändert
        """
        self._cq_load_errors(data_path, dataset_name)
        return super().load_cell_line_features(data_path, dataset_name)

    def _cq_load_errors(self, data_path: str, dataset_name: str) -> None:
        """Liest ``pEC50Error`` je (Zelllinie, Drug) aus der Datensatz-CSV in ``_ERROR_TABLES``.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :raises FileNotFoundError: wenn die Datensatz-CSV fehlt
        :raises ValueError: wenn ihr die Spalte ``pEC50Error`` fehlt
        """
        key = (os.path.abspath(data_path), dataset_name)
        if key in _ERROR_TABLES:
            return
        path = os.path.join(data_path, dataset_name, f"{dataset_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[CurveWeight] {path} fehlt — ohne Kurvenfehler ist dieses Modell nicht definiert")
        try:
            table = pd.read_csv(path, usecols=_QUALITY_FILE_COLS, low_memory=False)
        except ValueError as err:
            raise ValueError(
                f"[CurveWeight] {path} hat keine Spalte pEC50Error — ohne sie ist dieses Modell nicht definiert"
            ) from err
        table = table.dropna(subset=["pEC50Error"]).drop_duplicates(["cell_line_name", "pubchem_id"])
        _ERROR_TABLES[key] = {
            (str(c), str(d)): float(e)
            for c, d, e in zip(table.cell_line_name, table.pubchem_id, table.pEC50Error, strict=True)
        }
        print(f"[CurveWeight] {len(_ERROR_TABLES[key])} Kurvenfehler aus {os.path.basename(path)} geladen")

    @staticmethod
    def _cq_errors_for(dataset_name: str | None) -> dict[tuple[str, str], float]:
        """Fehlertabelle des Datensatzes aus dem modulweiten Cache.

        :param dataset_name: Name des Datensatzes
        :returns: Abbildung (Zelllinie, Drug) -> ``pEC50Error``
        :raises RuntimeError: wenn für den Datensatz nichts geladen wurde
        """
        for (_, name), table in _ERROR_TABLES.items():
            if name == dataset_name:
                return table
        raise RuntimeError(
            f"[CurveWeight] keine Kurvenfehler für Datensatz {dataset_name!r} geladen — "
            "das Training wäre unbemerkt ungewichtet gelaufen"
        )

    def _cq_weights(self, output: DrugResponseDataset) -> np.ndarray:
        """Gewichte für die Trainingszeilen; Mittelwert auf 1 normiert.

        :param output: Trainingsdatensatz
        :returns: Gewichtsvektor der Länge ``len(output)``
        :raises RuntimeError: wenn keine Trainingszeile einen Kurvenfehler bekommt
        """
        se_table = self._cq_errors_for(output.dataset_name)
        se_pec50 = np.array(
            [se_table.get((str(c), str(d)), np.nan) for c, d in zip(output.cell_line_ids, output.drug_ids)],
            dtype=float,
        )
        known = np.isfinite(se_pec50)
        if not known.any():
            raise RuntimeError("[CurveWeight] keine Trainingszeile konnte einer Kurve zugeordnet werden")

        if self._cq_mode == "trim":
            weights = (se_pec50 < self._cq_bad).astype(float)
            weights[~known] = 1.0  # ohne Fehlerangabe nicht wegwerfen
        else:
            se_response = self._cq_slope * np.minimum(se_pec50, self._cq_bad)  # jenseits der Schwelle gleich schlecht
            weights = 1.0 / (self._cq_tau**2 + se_response**2)
            weights[~known] = np.median(weights[known])

        total = weights.sum()
        if total <= 0:
            raise RuntimeError("[CurveWeight] alle Gewichte 0 — der Schwellwert wirft den ganzen Trainingssatz weg")
        kept = int((weights > 0).sum())
        print(
            f"[CurveWeight] Modus {self._cq_mode}: {len(weights)} Zeilen, {kept} mit Gewicht > 0 "
            f"({100.0 * kept / len(weights):.1f} %), {int((~known).sum())} ohne Fehlerangabe"
        )
        return weights * (len(weights) / total)

    def _fit_kwargs(self, output: DrugResponseDataset) -> dict:
        """Reicht das Gewicht als ``sample_weight`` an ``fit`` durch.

        :param output: Trainingsdatensatz (bereits transformiert)
        :returns: ``{"sample_weight": ...}``
        """
        return {"sample_weight": self._cq_weights(output)}


class RandomForestAEMolGNetMutCW(_CurveQualityWeightMixin, RandomForestAEMolGNetMut):
    """Wie ``RandomForestAEMolGNetMut``, aber inverse-varianz-gewichtet nach ``pEC50Error``."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutCW"""
        return "RandomForestAEMolGNetMutCW"


class RandomForestAEMolGNetMutTrim(_CurveQualityWeightMixin, RandomForestAEMolGNetMut):
    """Wie ``RandomForestAEMolGNetMut``, aber unbestimmbare Kurven fallen ganz aus dem Fit."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutTrim"""
        return "RandomForestAEMolGNetMutTrim"
