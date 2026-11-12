import pytest

from fraud.data import make_synthetic
from fraud.train import run


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(n_customers=300, n_terminals=500, tx_per_day=250, seed=7)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, synthetic_df):
    root = tmp_path_factory.mktemp("artifact")
    run(synthetic_df, root / "model", root / "reports", data_source="teste", synthetic=True)
    return root / "model"
