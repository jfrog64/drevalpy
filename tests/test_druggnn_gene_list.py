"""Tests for the configurable gene list of the DrugGNN model."""

import json

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.DrugGNN import drug_gnn
from drevalpy.models.DrugGNN.drug_gnn import DrugGNN

_GENE_EXPRESSION = (
    "cellosaurus_id,cell_line_name,TSPAN6,TNMD,BRCA1,SCYL3,HDAC1,INSIG1,FOXO3\n"
    "CVCL_1104,CAL-120,7.63,2.96,10.38,3.61,3.38,7.09,3.02\n"
    "CVCL_1174,DMS 114,7.55,2.78,11.81,4.07,3.73,2.80,6.08\n"
    "CVCL_1110,CAL-51,8.71,2.64,9.88,3.96,3.24,11.39,4.22\n"
)
#: Symbols of the two gene lists written by the gene_list_data_dir fixture.
_LANDMARK_GENES = ["TSPAN6", "BRCA1", "FOXO3"]
_OTHER_GENES = ["BRCA1", "SCYL3", "HDAC1", "INSIG1"]
_ALL_GENES = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3"]


@pytest.fixture
def gene_list_data_dir(tmp_path) -> str:
    """
    Build a minimal data directory with everything DrugGNN's cell line loader reads.

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
    (gene_lists / "gene_expression_intersection.csv").write_text(
        "Symbol\n" + "\n".join(_OTHER_GENES) + "\n", encoding="utf-8"
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


def _build(hyperparameters: dict | None = None) -> DrugGNN:
    """
    Instantiate and build a DrugGNN model.

    The gene list is independent of the network hyperparameters, and build_model only stores
    them, so an empty set is enough here.

    :param hyperparameters: hyperparameters to build with
    :returns: the built model
    """
    model = DrugGNN()
    model.build_model(hyperparameters=dict(hyperparameters or {}))
    return model


def test_druggnn_gene_list_default() -> None:
    """Without the hyperparameter the model keeps the previously hard-coded gene list."""
    assert MODEL_FACTORY["DrugGNN"] is DrugGNN
    assert DrugGNN.gene_list == "landmark_genes_reduced"
    assert _build().gene_list == "landmark_genes_reduced"


def test_druggnn_default_gene_list_is_used_when_loading(gene_list_data_dir) -> None:
    """
    The default has to reproduce the behaviour before the gene list became configurable.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build().load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_LANDMARK_GENES)


def test_druggnn_gene_list_hyperparameter(gene_list_data_dir) -> None:
    """
    The "gene_list" hyperparameter has to reach load_cell_line_features.

    :param gene_list_data_dir: path to the temporary data directory
    """
    model = _build({"gene_list": "gene_expression_intersection"})
    assert model.gene_list == "gene_expression_intersection"
    # It stays in the hyperparameters, so save_model()/load_model() carry it over to predict().
    assert model.hyperparameters["gene_list"] == "gene_expression_intersection"

    features = model.load_cell_line_features(data_path=gene_list_data_dir, dataset_name="GDSC1_small")
    assert sorted(_gene_names(features)) == sorted(_OTHER_GENES)
    # The class default is untouched, only the instance was reconfigured.
    assert DrugGNN.gene_list == "landmark_genes_reduced"


def test_druggnn_gene_list_none_loads_all_genes(gene_list_data_dir) -> None:
    """
    gene_list=None has to load the full expression matrix.

    :param gene_list_data_dir: path to the temporary data directory
    """
    features = _build({"gene_list": None}).load_cell_line_features(
        data_path=gene_list_data_dir, dataset_name="GDSC1_small"
    )
    assert sorted(_gene_names(features)) == sorted(_ALL_GENES)


def test_druggnn_gene_list_does_not_leak_between_instances(gene_list_data_dir) -> None:
    """
    A second model built without the hyperparameter must fall back to the default again.

    :param gene_list_data_dir: path to the temporary data directory
    """
    configured = _build({"gene_list": "gene_expression_intersection"})
    assert sorted(_gene_names(configured.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _OTHER_GENES
    )

    default = _build()
    assert sorted(_gene_names(default.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _LANDMARK_GENES
    )


def test_druggnn_gene_list_survives_the_save_load_round_trip(gene_list_data_dir, tmp_path, monkeypatch) -> None:
    """
    load_model() restores the config without calling build_model, so it must set the gene list itself.

    Otherwise the loaded model would predict on the class default gene space instead of the one it
    was trained on. Restoring the Lightning checkpoint is not what is under test here, so
    load_from_checkpoint is stubbed out and only the config file is written.

    :param gene_list_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    trained = _build({"gene_list": "gene_expression_intersection", "num_node_features": 1, "num_cell_features": 1})

    directory = tmp_path / "model"
    directory.mkdir()
    # This is what save_model() writes next to the checkpoint.
    (directory / "config.json").write_text(json.dumps(trained.hyperparameters), encoding="utf-8")
    monkeypatch.setattr(drug_gnn.DrugGNNModule, "load_from_checkpoint", classmethod(lambda cls, *a, **kw: None))

    restored = DrugGNN()
    restored.load_model(directory)

    assert restored.gene_list == "gene_expression_intersection"
    assert sorted(_gene_names(restored.load_cell_line_features(gene_list_data_dir, "GDSC1_small"))) == sorted(
        _OTHER_GENES
    )
