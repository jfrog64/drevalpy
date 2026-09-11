"""Buckley-James: zensierte Ziele iterativ nachführen — modellagnostisch, ohne zensierten Verlust.

Ausgangslage (gemessen 2026-09-03): 13,9 % der CTRPv2-Zeilen tragen einen ``LN_IC50`` **oberhalb**
der höchsten getesteten Dosis. Der eingetragene Wert ist dort eine ``curvecurator``-Extrapolation
(gedeckelt bei +2,30 ln-Einheiten = Faktor 10), die Aussage „IC50 > höchste Dosis" dagegen echt.
Das Kappen ``y = min(LN_IC50, ln(max_dose_M · 1e6))`` (Lauf ``gp20260903c_capped``) hat dafür
+0,0128 nPCC gebracht, 5/5 Splits.

Kappen ist formal die **erste** Buckley-James-Iteration: der zensierte Wert wird durch seine untere
Schranke ersetzt. Dieses Modul führt die Iteration weiter — je Runde

1. Modell auf den aktuellen Zielen fitten,
2. für die zensierten Zeilen eine **ehrliche** Vorhersage holen,
3. Ziel dieser Zeilen auf ``max(Schranke, Vorhersage)`` setzen,
4. neu fitten.

Damit lernt das Modell für „wirkt hier nicht"-Zeilen nicht mehr den harten Deckel, sondern die
eigene, plausiblere Fortschreibung nach oben — und bleibt trotzdem bei mindestens der Schranke,
weil genau das die Messung hergibt. Der Ansatz braucht keinen zensierten Verlust und läuft deshalb
auch mit RandomForest.

**Warum in-fold und nicht über die Out-of-fold-Vorhersagen des Referenzlaufs.** Naheliegend wäre,
die fertigen ``predictions_split_k.csv`` zu nehmen. Das leckt: die OOF-Vorhersage einer
Trainingszeile von Fold k stammt aus dem Modell **ihres eigenen** Folds j≠k, und dieses Modell hat
die Testzelllinien von Fold k im Training gesehen. Über das nachgeführte Ziel flösse damit
Testinformation in Fold k. Bei einem erwarteten Effekt in der Größenordnung 0,01 nPCC wäre das
nicht mehr von echtem Gewinn zu unterscheiden. Deshalb läuft die Iteration komplett **innerhalb**
eines Folds, im Modell selbst.

**Warum Out-of-bag statt In-sample.** In-sample-Vorhersagen eines RandomForest reproduzieren die
Trainingsziele nahezu exakt (die Bäume merken sich die Zeilen); ``max(Schranke, Vorhersage)`` wäre
dann ≈ das alte Ziel und die Iteration liefe leer. Die OOB-Vorhersage einer Zeile stammt dagegen
nur aus den Bäumen, die sie nicht im Bootstrap hatten — ehrlich und trotzdem fold-intern. Bei
``max_samples`` 0,2 ist jede Zeile in ~80 % der Bäume out-of-bag, die OOB-Schätzung ist also gut
gedeckt. Ohne OOB-Fähigkeit des Schätzers bricht das Modell ab statt still in-sample zu rechnen.

**Verhältnis zur Response-Transformation.** Mit ``--response_transformation drug_mean`` bekommt
``train`` bereits zentrierte Ziele; die Schranke steht aber in LN_IC50-Einheiten. Sie wird deshalb
je Zeile in denselben Raum geschoben (siehe :meth:`_BuckleyJamesMixin._bj_prepare`). Dass die
Transformation dafür rein additiv ist, wird geprüft, nicht angenommen.

``bj_iterations = 1`` ist exakt das bisherige Verhalten auf gekappten Splits — der Selbsttest gegen
``gp20260903c_capped``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .dipk_feature_sklearn import RandomForestAEMolGNetMut

#: Spalten der Datensatz-CSV, aus denen Schranke und Zensierung folgen.
_BOUND_FILE_COLS = ["cell_line_name", "pubchem_id", "max_dose_M", "LN_IC50_curvecurator"]

#: Schrankentabellen je (Datenpfad, Datensatz) — **modulweit, nicht je Modellinstanz**.
#:
#: Warum nicht auf ``self``: ``experiment._load_features_cached`` memoisiert
#: ``load_cell_line_features`` über ``_FEATURE_CACHE``. Ab dem dritten Fold ist der Cache-Key
#: stabil, die Lade-Methode wird also nicht mehr aufgerufen — während je Fold eine frische
#: Modellinstanz gebaut wird. Eine Instanzvariable wäre dann leer (genau der Fehler im Lauf
#: ``gp20260903_curveweight``, Folds 3-5). Die Tabelle hängt am Datensatz, nicht am Modell.
_BOUND_TABLES: dict[tuple[str, str], dict[tuple[str, str], tuple[float, float]]] = {}


class _BuckleyJamesMixin:
    """Wiederholt den Fit eines :class:`SklearnModel` mit nachgeführten zensierten Zielen.

    Hyperparameter:

    ``bj_iterations``
        Anzahl der Fits je Fold. 1 = keine Nachführung (identisch zum gekappten Basismodell).
    ``bj_update_source``
        ``oob`` (Vorgabe) oder ``insample`` — Quelle der Vorhersage für die Nachführung.
    ``bj_tolerance``
        Iteration abbrechen, sobald sich kein zensiertes Ziel um mehr als diesen Betrag
        (ln-Einheiten) ändert.
    """

    def build_model(self, hyperparameters: dict) -> None:
        """Basismodell bauen, BJ-Parameter übernehmen und den Schätzer OOB-fähig machen.

        :param hyperparameters: Hyperparameter des Modells
        :raises ValueError: bei unbekannter Quelle, unzulässiger Iterationszahl oder wenn der
            Schätzer keine Out-of-bag-Vorhersagen liefern kann
        """
        super().build_model(hyperparameters)
        self._bj_iterations = int(hyperparameters.get("bj_iterations", 3))
        self._bj_source = str(hyperparameters.get("bj_update_source", "oob"))
        self._bj_tol = float(hyperparameters.get("bj_tolerance", 0.001))
        if self._bj_iterations < 1:
            raise ValueError(f"bj_iterations muss >= 1 sein, nicht {self._bj_iterations}")
        if self._bj_source not in ("oob", "insample"):
            raise ValueError(f"bj_update_source muss 'oob' oder 'insample' sein, nicht {self._bj_source!r}")
        if self._bj_source == "oob" and self._bj_iterations > 1:
            params = self.model.get_params() if hasattr(self.model, "get_params") else {}
            if "oob_score" not in params or "bootstrap" not in params:
                raise ValueError(
                    f"[BJ] {type(self.model).__name__} kennt kein oob_score — mit in-sample-Vorhersagen "
                    "liefe die Iteration wirkungslos ins Leere, deshalb Abbruch"
                )
            self.model.set_params(oob_score=True, bootstrap=True)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Zelllinien-Features laden und dabei die Zensierungsschranken einlesen.

        Der Haken sitzt hier, weil ``train`` den Datenpfad nicht kennt, diese Methode aber schon.
        Sie läuft nicht je Fold (der Aufrufer memoisiert sie), deshalb der modulweite Cache.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset des Basismodells, unverändert
        """
        self._bj_load_bounds(data_path, dataset_name)
        return super().load_cell_line_features(data_path, dataset_name)

    @staticmethod
    def _bj_load_bounds(data_path: str, dataset_name: str) -> None:
        """Liest ``ln(max_dose_M · 1e6)`` und ``LN_IC50`` je (Zelllinie, Drug) in ``_BOUND_TABLES``.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :raises FileNotFoundError: wenn die Datensatz-CSV fehlt
        :raises ValueError: wenn ihr eine der benötigten Spalten fehlt
        """
        key = (os.path.abspath(data_path), dataset_name)
        if key in _BOUND_TABLES:
            return
        path = os.path.join(data_path, dataset_name, f"{dataset_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[BJ] {path} fehlt — ohne Dosisschranken ist dieses Modell nicht definiert")
        try:
            table = pd.read_csv(path, usecols=_BOUND_FILE_COLS, dtype={"pubchem_id": str}, low_memory=False)
        except ValueError as err:
            raise ValueError(
                f"[BJ] {path} fehlt eine der Spalten {_BOUND_FILE_COLS} — ohne sie ist das Modell nicht definiert"
            ) from err
        table = table.dropna(subset=["max_dose_M", "LN_IC50_curvecurator"]).drop_duplicates(
            ["cell_line_name", "pubchem_id"]
        )
        # Identische Rechnung wie in prepare_capped_splits_20260903.py -> bitgleiche Schranke.
        schranke = np.log(table.max_dose_M.to_numpy(dtype=float) * 1e6)
        _BOUND_TABLES[key] = {
            (str(c), str(d)): (float(b), float(y))
            for c, d, b, y in zip(table.cell_line_name, table.pubchem_id, schranke, table.LN_IC50_curvecurator)
        }
        zensiert = float(np.mean(table.LN_IC50_curvecurator.to_numpy(dtype=float) > schranke))
        print(
            f"[BJ] {len(_BOUND_TABLES[key])} Dosisschranken aus {os.path.basename(path)} geladen "
            f"({100 * zensiert:.1f} % der Zeilen zensiert)"
        )

    @staticmethod
    def _bj_bounds_for(dataset_name: str | None) -> dict[tuple[str, str], tuple[float, float]]:
        """Schrankentabelle des Datensatzes aus dem modulweiten Cache.

        :param dataset_name: Name des Datensatzes
        :returns: Abbildung (Zelllinie, Drug) -> (Schranke, LN_IC50)
        :raises RuntimeError: wenn für den Datensatz nichts geladen wurde
        """
        for (_, name), table in _BOUND_TABLES.items():
            if name == dataset_name:
                return table
        raise RuntimeError(
            f"[BJ] keine Dosisschranken für Datensatz {dataset_name!r} geladen — "
            "das Training wäre unbemerkt ohne Nachführung gelaufen"
        )

    def _bj_prepare(self, output: DrugResponseDataset) -> tuple[np.ndarray, np.ndarray]:
        """Schranke im Zielraum des Trainings und Zensierungsmarke je Trainingszeile.

        Die Schranke steht in LN_IC50-Einheiten, ``output.response`` kann aber transformiert sein
        (``drug_mean``). Der Versatz je Zeile wird deshalb aus den Daten selbst bestimmt:
        ``versatz = y_gekappt - y_transformiert``, wobei ``y_gekappt = min(LN_IC50, Schranke)``
        exakt der Wert ist, den ``prepare_capped_splits_20260903.py`` in die Split-Dateien
        geschrieben hat. Ist die Transformation rein additiv je Gruppe — nur dann ist eine
        Schranke überhaupt übertragbar —, ist dieser Versatz innerhalb eines Drugs konstant.
        Genau das wird geprüft: weicht eine Zeile ab, bricht das Training ab, statt eine falsche
        Schranke zu verwenden. Damit fällt auch der Fall auf, dass versehentlich **ungekappte**
        Splits verwendet wurden.

        :param output: Trainingsdatensatz (bereits transformiert)
        :returns: (Schranke im Zielraum, Maske der zensierten Zeilen)
        :raises RuntimeError: wenn Trainingszeilen keine Schranke haben oder der Versatz innerhalb
            eines Drugs nicht konstant ist
        """
        tabelle = self._bj_bounds_for(output.dataset_name)
        zeilen = [tabelle.get((str(c), str(d))) for c, d in zip(output.cell_line_ids, output.drug_ids)]
        fehlend = [i for i, z in enumerate(zeilen) if z is None]
        if fehlend:
            raise RuntimeError(
                f"[BJ] {len(fehlend)} von {len(zeilen)} Trainingszeilen ohne Dosisschranke "
                "— die Nachführung wäre für sie undefiniert"
            )
        schranke = np.array([z[0] for z in zeilen], dtype=float)
        ln_ic50 = np.array([z[1] for z in zeilen], dtype=float)
        y_gekappt = np.minimum(ln_ic50, schranke)
        y_ziel = np.asarray(output.response, dtype=float)

        versatz = y_gekappt - y_ziel
        drugs = np.asarray(output.drug_ids).astype(str)
        keys, inverse = np.unique(drugs, return_inverse=True)
        je_drug = np.zeros(keys.size, dtype=float)
        for i in range(keys.size):
            je_drug[i] = np.median(versatz[inverse == i])
        versatz_zeile = je_drug[inverse]
        abweichung = np.abs(versatz - versatz_zeile)
        schlimmste = int(np.argmax(abweichung))
        if abweichung[schlimmste] > 1e-6:
            raise RuntimeError(
                f"[BJ] Versatz zwischen Datensatz-CSV und Trainingszielen ist innerhalb von Drug "
                f"{str(drugs[schlimmste])!r} nicht konstant (max. Abweichung {abweichung[schlimmste]:.4g}). "
                "Entweder sind die Splits nicht gekappt oder die Response-Transformation ist nicht "
                "rein additiv — in beiden Fällen wäre die Schranke im Zielraum falsch."
            )

        zensiert = ln_ic50 > schranke
        print(
            f"[BJ] {int(zensiert.sum())} von {len(zensiert)} Trainingszeilen zensiert "
            f"({100 * zensiert.mean():.1f} %), Versatz je Drug im Median {np.median(je_drug):.4f}"
        )
        return schranke - versatz_zeile, zensiert

    def _bj_predictions(self, x: np.ndarray) -> np.ndarray:
        """Ehrliche Vorhersage der Trainingszeilen aus dem zuletzt gefitteten Modell.

        :param x: Designmatrix der Trainingszeilen
        :returns: Vorhersagen, ``NaN`` wo keine OOB-Schätzung existiert
        :raises RuntimeError: wenn OOB verlangt, aber nicht verfügbar ist
        """
        if self._bj_source == "insample":
            return np.asarray(self.model.predict(x), dtype=float)
        oob = getattr(self.model, "oob_prediction_", None)
        if oob is None:
            raise RuntimeError("[BJ] oob_prediction_ fehlt nach dem Fit — Nachführung nicht möglich")
        return np.asarray(oob, dtype=float)

    def _fit_estimator(self, x: np.ndarray, output: DrugResponseDataset) -> None:
        """Fit, dann ``bj_iterations - 1`` Runden mit nachgeführten zensierten Zielen.

        :param x: Designmatrix der Trainingszeilen
        :param output: Trainingsdatensatz (bereits transformiert)
        """
        kwargs = self._fit_kwargs(output)
        y = np.asarray(output.response, dtype=float).copy()
        # Schranken VOR dem ersten Fit pruefen: ein Abbruch wegen ungekappter Splits oder
        # nicht-additiver Transformation soll nicht erst nach einem vollen RF-Fit auffallen.
        schranke, zensiert = (None, None)
        if self._bj_iterations > 1:
            schranke, zensiert = self._bj_prepare(output)

        self.model.fit(x, y, **kwargs)
        if self._bj_iterations <= 1:
            return
        if not zensiert.any():
            print("[BJ] keine zensierte Trainingszeile — keine weitere Iteration")
            return

        for runde in range(2, self._bj_iterations + 1):
            vorhersage = self._bj_predictions(x)
            brauchbar = zensiert & np.isfinite(vorhersage)
            y_neu = y.copy()
            y_neu[brauchbar] = np.maximum(schranke[brauchbar], vorhersage[brauchbar])
            aenderung = np.abs(y_neu - y)[brauchbar]
            angehoben = int((y_neu[brauchbar] > y[brauchbar] + 1e-9).sum())
            print(
                f"[BJ] Iteration {runde}: {int(brauchbar.sum())} zensierte Zeilen nachgeführt, "
                f"{angehoben} davon angehoben, Änderung Median {np.median(aenderung):.4f} / "
                f"max {aenderung.max():.4f} ln-Einheiten"
                + (f", {int((zensiert & ~brauchbar).sum())} ohne OOB-Schätzung" if (zensiert & ~brauchbar).any() else "")
            )
            if aenderung.max() < self._bj_tol:
                print(f"[BJ] konvergiert (< {self._bj_tol} ln-Einheiten) — Abbruch vor Iteration {runde}")
                return
            y = y_neu
            self.model.fit(x, y, **kwargs)


class RandomForestAEMolGNetMutBJ(_BuckleyJamesMixin, RandomForestAEMolGNetMut):
    """Wie ``RandomForestAEMolGNetMut``, aber mit weiteren Buckley-James-Iterationen."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutBJ"""
        return "RandomForestAEMolGNetMutBJ"
