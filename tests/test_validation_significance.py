import numpy as np
import pandas as pd

from src.evaluation.validation_significance import (
    blockwise_nested_deviance_f_test,
    information_criteria_report,
)


def test_blockwise_nested_deviance_f_test_identifies_signal_block():
    rng = np.random.default_rng(11)
    n_train = 320
    n_test = 120

    signal_train = rng.normal(0.0, 1.0, n_train)
    noise_train = rng.normal(0.0, 1.0, n_train)
    p_train = 1.0 / (1.0 + np.exp(-(1.8 * signal_train + 0.1 * noise_train)))
    y_train = rng.binomial(1, p_train)

    signal_test = rng.normal(0.0, 1.0, n_test)
    noise_test = rng.normal(0.0, 1.0, n_test)
    p_test = 1.0 / (1.0 + np.exp(-(1.8 * signal_test + 0.1 * noise_test)))
    y_test = rng.binomial(1, p_test)

    train_df = pd.DataFrame({"home_win": y_train, "signal": signal_train, "noise": noise_train})
    test_df = pd.DataFrame({"home_win": y_test, "signal": signal_test, "noise": noise_test})

    report = blockwise_nested_deviance_f_test(
        train_df,
        test_df,
        feature_blocks={"signal_block": ["signal"], "noise_block": ["noise"]},
        all_features=["signal", "noise"],
    )

    assert {"f_stat", "p_value", "deviance_drop", "delta_log_loss", "delta_brier"}.issubset(report.columns)
    signal_row = report.set_index("block").loc["signal_block"]
    noise_row = report.set_index("block").loc["noise_block"]
    assert signal_row["f_stat"] > noise_row["f_stat"]
    assert signal_row["delta_log_loss"] > noise_row["delta_log_loss"]


def test_information_criteria_report_returns_candidate_table():
    rng = np.random.default_rng(23)
    n = 180
    signal = rng.normal(0.0, 1.0, n)
    extra = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(1.4 * signal + 0.2 * extra)))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"home_win": y, "signal": signal, "extra": extra})

    report = information_criteria_report(
        df.iloc[:120].copy(),
        df.iloc[120:].copy(),
        feature_blocks={"signal_block": ["signal"], "extra_block": ["extra"]},
        all_features=["signal", "extra"],
    )

    candidates = report["candidates"]
    summary = report["summary"]

    assert not candidates.empty
    assert {"candidate", "aic", "bic", "holdout_log_loss"}.issubset(candidates.columns)
    assert summary["status"] == "ok"
    assert summary["candidate_count"] >= 2
