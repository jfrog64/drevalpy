"""Drug-Level-Trim: ganze Substanzen jenseits der Rausch-Kante aus dem TRAINING werfen.

Abgrenzung zu ``curve_quality_weights`` (wichtig, sonst wird das hier fuer schon getestet gehalten):
Die dortigen Arme ``…CW`` und ``…Trim`` arbeiten **pointwise** — jede einzelne Zeile mit
``pEC50Error >= 1,0`` wird abgewertet bzw. auf Gewicht 0 gesetzt, 81,5 % der Zeilen bleiben. Beide
sind gemessen (2026-09-03, Lauf ``gp20260903b_curveweight``): CW nPCC 0,2535 gegen 0,3372
ungewichtet (0/5 Splits besser), Trim 0,3079. Genau das sagt die Rauschanalyse vom selben Tag
voraus — das Rauschen ist **drug**-spezifisch, und fuers Modell ist es eine **Kante, keine Rampe**:
Median within-drug rho 0,414 / 0,417 / 0,424 bis 50 % unbestimmbarer Kurven je Drug, dann 0,285
(50–90 %) und 0,004 (>90 %). Solange ein Drug genug brauchbare Zelllinien behaelt, kostet
Punktgewichtung nur effektive Stichprobe.

Dieses Modul testet deshalb die passende Operation: die ~131 Substanzen jenseits der Kante
**ganz** aus dem Trainingssatz nehmen. Bei 23,8 % der Drugs liegt der Median-IC50 ueber der
hoechsten getesteten Konzentration; diese Zeilen sind extrapoliert, nicht gemessen (Spearman 0,87
zwischen Rauschanteil und ``ln(Median-IC50) - ln(top_test_conc)``). Frage: lernt das Modell besser,
wenn es sie gar nicht erst sieht?

Drei Festlegungen, an denen der Vergleich haengt:

1. **Nur das Training wird beschnitten, der Testsatz bleibt vollstaendig.** Sonst waere der
   Vergleich gegen die Referenz ein Massstabswechsel und kein Modellunterschied — derselbe
   Fallstrick, an dem der pointwise ``…Trim``-Arm auf der Stufe ``punkt`` scheinbar gewann.
2. **Der Rauschanteil je Drug wird aus dem TRAININGS-Fold berechnet**, nicht aus der ganzen
   Datensatz-CSV. Die Fehlerspalte stammt aus demselben Assay wie die Zielwerte; ueber alle
   Zelllinien gemittelt liefe eine (milde) Leckage aus den Test-Zelllinien mit.
3. **Die Zeilen fliegen wirklich raus, sie bekommen nicht Gewicht 0.** Bei ``max_samples=0.2``
   zoege eine Zeile mit Gewicht 0 trotzdem einen Platz in der Teilstichprobe jedes Baums und
   senkte damit die effektive Stichprobengroesse — der Unterschied waere nicht mehr sauber
   "Substanz weg" gegen "Substanz drin".

**Kein stiller Rueckfall.** Fehlt die Fehlertabelle, bricht das Training ab. Lehre aus dem
ungueltigen Lauf ``gp20260903_curveweight``: ein Arm, der unbemerkt wie das Basismodell trainiert,
sieht aus wie ein Ergebnis. Die Fehlertabelle wird deshalb aus dem modulweiten Cache von
``curve_quality_weights`` gelesen (dort liegt sie, weil ``load_cell_line_features`` vom Aufrufer
memoisiert wird und eine Instanzvariable ab Fold 3 leer waere).

**Was mit den ausgeworfenen Substanzen im Test passiert:** ``experiment.train_and_predict`` fittet
die Response-Transformation (``drug_mean``) auf dem **vollstaendigen** Trainingssatz, bevor
``train()`` aufgerufen wird. Der Drug-Mittelwert einer beschnittenen Substanz existiert also
weiterhin; sie wird im Test vorhergesagt, das Modell hat fuer sie nur keine eigenen Trainingszeilen
mehr. Genau das ist die Frage des Arms.
"""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .curve_quality_weights import _CurveQualityWeightMixin as _CQ
from .dipk_feature_sklearn import RandomForestAEMolGNetMut


class _DrugLevelTrimMixin:
    """Wirft vor dem Fit alle Trainingszeilen der Substanzen jenseits der Rausch-Kante weg."""

    def build_model(self, hyperparameters: dict) -> None:
        """Basismodell bauen und die Trim-Schwellen uebernehmen.

        :param hyperparameters: Hyperparameter des Modells
        :raises ValueError: wenn der Schwellenanteil nicht in (0, 1] liegt
        """
        super().build_model(hyperparameters)
        self._dt_bad = float(hyperparameters.get("drug_trim_bad_pec50_error", 1.0))
        self._dt_frac = float(hyperparameters.get("drug_trim_max_bad_frac", 0.5))
        self._dt_min = int(hyperparameters.get("drug_trim_min_curves", 10))
        if not 0.0 < self._dt_frac <= 1.0:
            raise ValueError(f"drug_trim_max_bad_frac muss in (0, 1] liegen, nicht {self._dt_frac}")

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Zelllinien-Features laden und dabei die Kurvenfehler in den modulweiten Cache holen.

        :param data_path: Pfad zu den Daten
        :param dataset_name: Name des Datensatzes
        :returns: FeatureDataset des Basismodells, unveraendert
        """
        _CQ._cq_load_errors(self, data_path, dataset_name)
        return super().load_cell_line_features(data_path, dataset_name)

    def _dt_keep_mask(self, output: DrugResponseDataset) -> np.ndarray:
        """Boolesche Maske der Trainingszeilen, die im Fit bleiben.

        :param output: Trainingsdatensatz
        :returns: Maske der Laenge ``len(output)``
        :raises RuntimeError: wenn keine Zeile einer Kurve zugeordnet werden kann
        """
        se_table = _CQ._cq_errors_for(output.dataset_name)
        se = np.array(
            [se_table.get((str(c), str(d)), np.nan) for c, d in zip(output.cell_line_ids, output.drug_ids)],
            dtype=float,
        )
        known = np.isfinite(se)
        if not known.any():
            raise RuntimeError("[DrugTrim] keine Trainingszeile konnte einer Kurve zugeordnet werden")

        drugs = np.asarray(output.drug_ids).astype(str)
        keep = np.ones(len(drugs), dtype=bool)
        raus, ohne_angabe = [], 0
        for drug in np.unique(drugs):
            zeilen = drugs == drug
            gemessen = zeilen & known
            n = int(gemessen.sum())
            if n == 0:
                ohne_angabe += 1  # ohne Fehlerangabe nicht wegwerfen
                continue
            anteil = float((se[gemessen] >= self._dt_bad).sum()) / n
            if n >= self._dt_min and anteil >= self._dt_frac:
                keep[zeilen] = False
                raus.append((drug, anteil, int(zeilen.sum())))

        if not keep.any():
            raise RuntimeError("[DrugTrim] die Schwelle wirft den ganzen Trainingssatz weg")
        print(
            f"[DrugTrim] Schwelle {self._dt_frac:.2f} (pEC50Error >= {self._dt_bad}): "
            f"{len(raus)} von {len(np.unique(drugs))} Substanzen raus, "
            f"{int((~keep).sum())} von {len(keep)} Zeilen ({100.0 * (~keep).mean():.1f} %), "
            f"{ohne_angabe} Substanzen ohne Fehlerangabe behalten"
        )
        return keep

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Trainingssatz beschneiden, dann normal weitertrainieren.

        :param output: Trainingsdaten (Response) -- wird **nicht** veraendert, es geht eine Kopie weiter
        :param cell_line_input: Zelllinien-Features
        :param drug_input: Drug-Features
        :param output_earlystopping: unveraendert durchgereicht
        :param model_checkpoint_dir: unveraendert durchgereicht
        :raises RuntimeError: wenn der Trim die Menge der Trainingszelllinien aendert
        """
        if len(output) > 0:
            keep = self._dt_keep_mask(output)
            beschnitten = output.copy()
            beschnitten.mask(keep)
            # Der AE-Encoder wird auf np.unique(cell_line_ids) gefittet und ueber genau diese Menge
            # im DREVAL_AE_CACHE geschluesselt. Bliebe eine Zelllinie auf der Strecke, saehe dieser
            # Arm einen anderen Encoder als die Referenz und der Unterschied waere nicht mehr allein
            # der Trim. In LCO/CTRPv2 kann das nicht passieren (jede Linie traegt hunderte Drugs) --
            # deshalb ist es hier eine Abbruchbedingung und keine Warnung.
            vorher = set(np.asarray(output.cell_line_ids).astype(str))
            nachher = set(np.asarray(beschnitten.cell_line_ids).astype(str))
            if vorher != nachher:
                raise RuntimeError(
                    f"[DrugTrim] der Trim entfernt {len(vorher - nachher)} Zelllinien komplett aus dem "
                    "Training -- der AE-Encoder waere ein anderer als in der Referenz, der Vergleich "
                    "damit nicht mehr sauber"
                )
            output = beschnitten
        super().train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
        )


class RandomForestAEMolGNetMutDrugTrim(_DrugLevelTrimMixin, RandomForestAEMolGNetMut):
    """Wie ``RandomForestAEMolGNetMut``, aber Substanzen jenseits der Rausch-Kante fehlen im Fit."""

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: RandomForestAEMolGNetMutDrugTrim"""
        return "RandomForestAEMolGNetMutDrugTrim"
