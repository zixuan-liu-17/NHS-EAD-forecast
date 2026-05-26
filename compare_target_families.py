from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_target_variable import load_daily_target

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import GammaRegressor, LinearRegression, PoissonRegressor, TweedieRegressor
    from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error, mean_tweedie_deviance
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
except ImportError as exc:  # pragma: no cover - environment-specific guard
    raise SystemExit(
        "This script requires scikit-learn. Install it with 'python3 -m pip install scikit-learn'."
    ) from exc


EPSILON = 1e-6
HORIZONS = list(range(1, 11))


def make_origin_frame(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame["target"] = series
    frame["lag_3"] = series.shift(3)
    frame["lag_4"] = series.shift(4)
    frame["lag_7"] = series.shift(7)
    frame["lag_10"] = series.shift(10)
    frame["lag_14"] = series.shift(14)
    frame["lag_21"] = series.shift(21)
    frame["roll_mean_7_d3"] = series.shift(3).rolling(7).mean()
    frame["roll_mean_28_d3"] = series.shift(3).rolling(28).mean()
    frame["roll_std_7_d3"] = series.shift(3).rolling(7).std()
    frame["roll_std_28_d3"] = series.shift(3).rolling(28).std()
    frame["roll_min_7_d3"] = series.shift(3).rolling(7).min()
    frame["roll_max_7_d3"] = series.shift(3).rolling(7).max()

    day_of_week = frame.index.dayofweek.to_numpy()
    day_of_year = frame.index.dayofyear.to_numpy()
    frame["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    frame["annual_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["annual_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["trend"] = np.arange(len(frame), dtype=float)

    return frame


def make_horizon_frame(origin_frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = origin_frame.copy()
    frame["target"] = origin_frame["target"].shift(-horizon)
    frame["forecast_origin"] = frame.index
    return frame.dropna().copy()


def make_models(tweedie_power: float) -> dict[str, Pipeline | TransformedTargetRegressor]:
    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    positive_transformer = FunctionTransformer(
        func=lambda values: np.log(np.clip(values, EPSILON, None)),
        inverse_func=np.exp,
        validate=False,
    )

    gaussian_log_model = TransformedTargetRegressor(
        regressor=linear_model,
        transformer=positive_transformer,
        check_inverse=False,
    )

    return {
        "Gaussian": gaussian_log_model,
        "Poisson": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
            ]
        ),
        "Gamma": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", GammaRegressor(alpha=0.01, max_iter=1000)),
            ]
        ),
        "Tweedie": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", TweedieRegressor(power=tweedie_power, alpha=0.01, link="log", max_iter=1000)),
            ]
        ),
    }


def tune_tweedie_power(data: pd.DataFrame, min_train_size: int, powers: list[float]) -> float:
    candidate_scores: dict[float, float] = {}
    validation_size = min(90, max(30, len(data) // 6))

    for power in powers:
        model = make_models(power)["Tweedie"]
        start_index = max(min_train_size, len(data) - validation_size)
        preds: list[float] = []
        actuals: list[float] = []

        for i in range(start_index, len(data)):
            train = data.iloc[:i]
            test = data.iloc[i : i + 1]
            model.fit(train.drop(columns=["target", "forecast_origin"]), train["target"])
            prediction = float(model.predict(test.drop(columns=["target", "forecast_origin"]))[0])
            preds.append(max(prediction, EPSILON))
            actuals.append(float(test["target"].iloc[0]))

        candidate_scores[power] = mean_tweedie_deviance(actuals, preds, power=power)

    return min(candidate_scores, key=candidate_scores.get)


def backtest_single_horizon(
    data: pd.DataFrame,
    min_train_size: int,
    tweedie_power: float,
    horizon: int,
) -> pd.DataFrame:
    models = make_models(tweedie_power)
    predictions: list[dict[str, float | str]] = []

    for i in range(min_train_size, len(data)):
        train = data.iloc[:i]
        test = data.iloc[i : i + 1]
        features_train = train.drop(columns="target")
        target_train = train["target"]
        features_test = test.drop(columns="target")
        actual = float(test["target"].iloc[0])

        row: dict[str, float | str] = {
            "forecast_origin": test["forecast_origin"].iloc[0].strftime("%Y-%m-%d"),
            "target_date": test.index[0].strftime("%Y-%m-%d"),
            "horizon": horizon,
            "actual": actual,
        }

        for name, model in models.items():
            model.fit(features_train.drop(columns="forecast_origin"), target_train)
            pred = float(model.predict(features_test.drop(columns="forecast_origin"))[0])
            row[name] = max(pred, EPSILON)

        predictions.append(row)

    return pd.DataFrame(predictions)


def summarise_metrics(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    family_names = ["Gaussian", "Poisson", "Gamma", "Tweedie"]
    metrics = []

    for name in family_names:
        predicted_values = prediction_frame[name].to_numpy()
        actual_values = prediction_frame["actual"].to_numpy()
        early_mask = prediction_frame["horizon"].between(1, 5).to_numpy()
        late_mask = prediction_frame["horizon"].between(6, 10).to_numpy()

        metrics.append(
            {
                "family": name,
                "mse_1_5": mean_squared_error(actual_values[early_mask], predicted_values[early_mask]),
                "mse_6_10": mean_squared_error(actual_values[late_mask], predicted_values[late_mask]),
                "rmse_all": mean_squared_error(actual_values, predicted_values) ** 0.5,
                "mae": mean_absolute_error(actual_values, predicted_values),
                "bias": float(np.mean(predicted_values - actual_values)),
                "corr": float(np.corrcoef(actual_values, predicted_values)[0, 1]),
                "poisson_deviance": mean_poisson_deviance(actual_values, predicted_values),
                "gamma_deviance": mean_gamma_deviance(actual_values, predicted_values),
                "tweedie_deviance_1_5": mean_tweedie_deviance(actual_values, predicted_values, power=1.5),
                "below_zero_share": float(np.mean(predicted_values <= 0)),
            }
        )

    metrics_frame = pd.DataFrame(metrics).sort_values(["mse_1_5", "mse_6_10"]).reset_index(drop=True)
    return metrics_frame


def rolling_backtest(data: pd.DataFrame, min_train_size: int, tweedie_power: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_predictions = []

    for horizon in HORIZONS:
        horizon_frame = make_horizon_frame(data, horizon)
        horizon_predictions.append(backtest_single_horizon(horizon_frame, min_train_size, tweedie_power, horizon))

    prediction_frame = pd.concat(horizon_predictions, ignore_index=True)
    metrics_frame = summarise_metrics(prediction_frame)
    return prediction_frame, metrics_frame


def choose_recommendation(metrics_frame: pd.DataFrame, series: pd.Series, tweedie_power: float) -> str:
    best_family = str(metrics_frame.iloc[0]["family"])
    best_mse_1_5 = float(metrics_frame.iloc[0]["mse_1_5"])
    best_mse_6_10 = float(metrics_frame.iloc[0]["mse_6_10"])
    gaussian_mse_1_5 = float(metrics_frame.loc[metrics_frame["family"] == "Gaussian", "mse_1_5"].iloc[0])
    gaussian_mse_6_10 = float(metrics_frame.loc[metrics_frame["family"] == "Gaussian", "mse_6_10"].iloc[0])
    dispersion = float(series.var(ddof=1) / series.mean())
    fractional_share = float(((series % 1).abs() > 1e-9).mean())

    lines = [
        f"Best D-3 compliant family by ranking: {best_family}.",
        f"Best MSE_1_5: {best_mse_1_5:.6f}.",
        f"Best MSE_6_10: {best_mse_6_10:.6f}.",
        f"Selected Tweedie power: {tweedie_power:.2f}.",
    ]

    if fractional_share > 0.5:
        lines.append("The target is fractional throughout, so Poisson, Gamma, and Tweedie should be interpreted as positive-mean modeling choices rather than literal count likelihoods.")

    if dispersion < 0.8:
        lines.append("The target is under-dispersed relative to Poisson, which weakens a plain Poisson story.")

    if best_family == "Gaussian" or (
        gaussian_mse_1_5 <= best_mse_1_5 * 1.02 and gaussian_mse_6_10 <= best_mse_6_10 * 1.02
    ):
        lines.append("Recommendation: start with a Gaussian-style baseline on a transformed positive target; it is the safest default when the label is fractional and not count-like.")
    elif best_family == "Gamma":
        lines.append("Recommendation: Gamma is the most defensible positive-target assumption here because the series is strictly positive and continuous.")
    elif best_family == "Tweedie":
        lines.append("Recommendation: Tweedie is a reasonable compromise when you want positive predictions and a variance function more flexible than Gamma or Poisson.")
    else:
        lines.append("Recommendation: Poisson only makes sense here as a log-link mean model, not as a literal count-data assumption.")

    return "\n".join(lines)


def save_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame, recommendation: str, output_dir: Path) -> None:
    predictions.to_csv(output_dir / "family_predictions.csv", index=False)
    metrics.to_csv(output_dir / "family_metrics.csv", index=False)
    (output_dir / "family_recommendation.txt").write_text(recommendation + "\n", encoding="utf-8")


def plot_scorecard(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_sorted = metrics.sort_values("mse_1_5")

    axes[0].bar(metrics_sorted["family"], metrics_sorted["mse_1_5"], color=["#355070", "#6d597a", "#b56576", "#e56b6f"])
    axes[0].set_title("D-3 Compliant MSE 1-5")
    axes[0].set_ylabel("MSE")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(metrics_sorted["family"], metrics_sorted["mse_6_10"], color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"])
    axes[1].set_title("D-3 Compliant MSE 6-10")
    axes[1].set_ylabel("MSE")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "family_scorecard.png", dpi=160)
    plt.close(fig)


def plot_prediction_panel(predictions: pd.DataFrame, output_dir: Path) -> None:
    focus = predictions.loc[predictions["horizon"].isin([1, 10])].copy()
    focus = focus.sort_values(["horizon", "forecast_origin"]).groupby("horizon").tail(120)
    focus_dates = pd.to_datetime(focus["forecast_origin"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    family_names = ["Gaussian", "Poisson", "Gamma", "Tweedie"]
    colors = {
        "Gaussian": "#355070",
        "Poisson": "#6d597a",
        "Gamma": "#b56576",
        "Tweedie": "#e56b6f",
    }

    for ax, family in zip(axes.ravel(), family_names):
        horizon_1 = focus.loc[focus["horizon"] == 1]
        horizon_10 = focus.loc[focus["horizon"] == 10]
        ax.plot(pd.to_datetime(horizon_1["forecast_origin"]), horizon_1["actual"], color="#111111", linewidth=1.5, label="Actual h=1")
        ax.plot(pd.to_datetime(horizon_1["forecast_origin"]), horizon_1[family], color=colors[family], linewidth=1.4, label=f"{family} h=1")
        ax.plot(pd.to_datetime(horizon_10["forecast_origin"]), horizon_10["actual"], color="#111111", linewidth=1.0, linestyle="--", label="Actual h=10")
        ax.plot(pd.to_datetime(horizon_10["forecast_origin"]), horizon_10[family], color=colors[family], linewidth=1.0, linestyle="--", label=f"{family} h=10")
        ax.set_title(family)
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "family_prediction_panel.png", dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    family_names = ["Gaussian", "Poisson", "Gamma", "Tweedie"]

    lower = min(predictions["actual"].min(), predictions[family_names].min().min())
    upper = max(predictions["actual"].max(), predictions[family_names].max().max())

    for ax, family in zip(axes.ravel(), family_names):
        ax.scatter(predictions["actual"], predictions[family], color="#355070", alpha=0.55, s=18)
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#bc4749")
        ax.set_title(family)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "family_actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def plot_horizon_profile(predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    family_names = ["Gaussian", "Poisson", "Gamma", "Tweedie"]
    colors = {
        "Gaussian": "#355070",
        "Poisson": "#6d597a",
        "Gamma": "#b56576",
        "Tweedie": "#e56b6f",
    }

    for family in family_names:
        horizon_mse = []
        for horizon in HORIZONS:
            subset = predictions.loc[predictions["horizon"] == horizon]
            horizon_mse.append(mean_squared_error(subset["actual"], subset[family]))
        ax.plot(HORIZONS, horizon_mse, marker="o", color=colors[family], label=family)

    ax.set_title("MSE by Forecast Horizon (D-3 compliant)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "family_horizon_profile.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gaussian, Poisson, Gamma, and Tweedie target assumptions.")
    parser.add_argument("--zip-path", default="data/turingAI_forecasting_challenge_dataset.csv.zip")
    parser.add_argument("--output-dir", default="report/family_comparison")
    parser.add_argument("--min-train-size", type=int, default=180)
    parser.add_argument("--tweedie-powers", default="1.1,1.3,1.5,1.7,1.9")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_target = load_daily_target(Path(args.zip_path))
    supervised = make_origin_frame(daily_target)
    tweedie_powers = [float(value) for value in args.tweedie_powers.split(",")]
    tweedie_tuning_frame = make_horizon_frame(supervised, 1)
    tweedie_power = tune_tweedie_power(tweedie_tuning_frame, args.min_train_size, tweedie_powers)
    predictions, metrics = rolling_backtest(supervised, args.min_train_size, tweedie_power)
    recommendation = choose_recommendation(metrics, daily_target, tweedie_power)

    save_outputs(predictions, metrics, recommendation, output_dir)
    plot_scorecard(metrics, output_dir)
    plot_prediction_panel(predictions, output_dir)
    plot_actual_vs_predicted(predictions, output_dir)
    plot_horizon_profile(predictions, output_dir)

    print(metrics.to_string(index=False))
    print()
    print(recommendation)
    print()
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()