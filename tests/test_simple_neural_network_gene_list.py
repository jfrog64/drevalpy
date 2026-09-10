"""Tests for the configurable gene list of the SimpleNeuralNetwork model."""

import json
from pathlib import Path

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

_GENE_EXPRESSION = (
    "cellosaurus_id,cell_line_name,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
    "CVCL_1104,CAL-120,7.63,2.96,10.38,3.61,3.38,7.09,3.02\n"
    "CVCL_1174,DMS 114,7.55,2.78,11.81,4.07,3.73,2.80,6.08\n"
    "CVCL_1110,CAL-51,8.71,2.64,9.88,3.96,3.24,11.39,4.22\n"
)
#: Symbols of the two gene lists written by the gene_list_data_dir fixture.
_LANDMARK_GENES = ["BRCA1", "SCYL3", "INSIG1", "FOXO3"]
_TARGET_GENES = ["TSPAN6", "SCYL3", "BRCA1"]
_ALL_GENES = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]


@pytest.fixture
def gene_list_data_dir(tmp_path) -> str:
    """
    Build a minimal data directory with a gene expression matrix and two gene lists.

    :param tmp_path: pytest temporary path fixture
    :returns: path to the data directory
    """
    dataset_dir = tmp_path / "GDSC1_small"
    dataset_dir.mkdir()
    (dataset_dir / "gene_expression.csv").write_text(_GENE_EXPRESSION, encoding="utf-8")

    gene_lists = tmp_path / "meta" / "gene_lists"
    gene_lists.mkdir(parents=True)
    (gene_lists / "landmark_genes_reduced.csv").write_text(
        "Symbol\n" + "\n".join(_LANDMARK_GENES) + "\n", encoding="utf-8"
    )
    (gene_lists / "drug_target_genes_all_drugs.csv").write_text(
        "Symbol\n" + "\n".join(_TARGET_GENES) + "\n", encoding="utf-8"
    )
    return str(tmp_path)


def _gene_names(features) -> list[str]:
    """
    Extract the gene symbols of the gene_expression view.

    :param features: FeatureDataset returned by a loader
    :returns: gene symbols
    """
    assert features.meta_info is not None
    return list(features.meta_info["gene_expression"])


def _build(hyperparameters: dict | None = None) -> SimpleNeuralNetwork:
    """
    Instantiate and build a SimpleNeuralNetwork.

    The gene list is independent of the network hyperparameters, and the network itself is only
    built in train(), so an empty set is enough here.

    :param hyperparameters: hyperparameters to build with
    :returns: the built model
    """
    model = SimpleNeuralNetwork()
    model.build_model(hyperparameters=dict(hyperparameters or {}))
    return model


def test_simple_neural_network_gene_list_default() -> None:
    """Without the hyperparameter the model keeps the previously hard-coded gene list."""
    assert MODEL_FACTORY["SimpleNeuralNetwork"] is SimpleNeuralNetwork
    assert SimpleNeuralNetwork.gene_list == "landmark_genes_reduced"
    assert _build().gene_list == "landmark_genes_reduced"


def test_simple_neural_network_default_gene_list_is_used_when_loading(gene_list_data_dir) -> None:
    """
    The default has to reproduce the behaviour before the gene list was passed through.

    Before this change the model called load_single_cell_line_view without gene_list, so the
    helper's own default applied -- which is the same list. The gene space must not move.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build().load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_LANDMARK_GENES)


def test_simple_neural_network_gene_list_hyperparameter(gene_list_data_dir) -> None:
    """
    The "gene_list" hyperparameter has to reach load_cell_line_features.

    :param gene_list_data_dir: path to the temporary data directory
    """
    model = _build({"gene_list": "drug_target_genes_all_drugs"})
    assert model.gene_list == "drug_target_genes_all_drugs"
    # It stays in the hyperparameters, so save()/load() carry it over to predict().
    assert model.hyperparameters["gene_list"] == "drug_target_genes_all_drugs"

    features = model.load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_TARGET_GENES)
    # The class default is untouched, only the instance was reconfigured.
    assert SimpleNeuralNetwork.gene_list == "landmark_genes_reduced"


def test_simple_neural_network_gene_list_none_loads_all_genes(gene_list_data_dir) -> None:
    """
    gene_list=None has to load the full expression matrix.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build({"gene_list": None}).load_cell_line_features(
        data_path=gene_list_data_dir, dataset_name="GDSC1_small"
    )
    assert sorted(_gene_names(features)) == sorted(_ALL_GENES)


def test_simple_neural_network_gene_list_only_touches_gene_expression(gene_list_data_dir) -> None:
    """
    A non gene_expression view is loaded whole, the gene list must not reach it.

    :param gene_list_data_dir: path to the temporary data directory
    """
    (Path(gene_list_data_dir) / "GDSC1_small" / "methylation.csv").write_text(
        "cellosaurus_id,cell_line_name,cg01,cg02\nCVCL_1104,CAL-120,0.1,0.2\n",
        encoding="utf-8",
    )
    model = _build({"cell_line_views": "methylation", "gene_list": "drug_target_genes_all_drugs"})
    features = model.load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")

    assert features.view_names == ["methylation"]
    # Both columns are still there, the gene list did not subset this view.
    assert len(features.features["CAL-120"]["methylation"]) == 2


def test_simple_neural_network_gene_list_does_not_leak_between_instances(gene_list_data_dir) -> None:
    """
    A second model built without the hyperparameter must fall back to the default again.

    :param gene_list_data_dir: path to the temporary data directory
    """
    configured = _build({"gene_list": "drug_target_genes_all_drugs"})
    assert sorted(_gene_names(configured.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _TARGET_GENES
    )

    default = _build()
    assert sorted(_gene_names(default.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _LANDMARK_GENES
    )


def test_simple_neural_network_gene_list_survives_the_save_load_round_trip(gene_list_data_dir) -> None:
    """
    load() rebuilds from the saved hyperparameters, so the restored model must use the same genes.

    :param gene_list_data_dir: path to the temporary data directory
    """
    trained = _build({"gene_list": "drug_target_genes_all_drugs"})
    # This is what save() writes and load() reads back before calling build_model().
    restored = SimpleNeuralNetwork()
    restored.build_model(json.loads(json.dumps(trained.hyperparameters)))

    assert restored.gene_list == "drug_target_genes_all_drugs"
    assert sorted(_gene_names(restored.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _TARGET_GENES
    )
