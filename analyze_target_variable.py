from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_METRIC = "estimated_avoidable_deaths"
DEVELOPMENT_END = pd.Timestamp("2025-09-30", tz="UTC")


def poisson_pmf(k_values: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0:
        probs = np.zeros_like(k_values, dtype=float)
        probs[0] = 1.0
        return probs

    log_rate = math.log(rate)
    return np.array(
        [math.exp(k * log_rate - rate - math.lgamma(k + 1)) for k in k_values],
        dtype=float,
    )


def fit_negative_binomial(mean_value: float, variance_value: float) -> tuple[float, float] | None:
    if mean_value <= 0 or variance_value <= mean_value:
        return None

    size = (mean_value ** 2) / (variance_value - mean_value)
    probability = size / (size + mean_value)
    return size, probability


def negative_binomial_pmf(k_values: np.ndarray, size: float, probability: float) -> np.ndarray:
    return np.array(
        [
            math.exp(
                math.lgamma(k + size)
                - math.lgamma(size)
                - math.lgamma(k + 1)
                + size * math.log(probability)
                + k * math.log(1 - probability)
            )
            for k in k_values
        ],
        dtype=float,
    )


def load_daily_target(zip_path: Path) -> pd.Series:
    df = pd.read_csv(
        zip_path,
        compression="zip",
        usecols=["dt", "metric_name", "value"],
    )

    target_df = df.loc[df["metric_name"] == TARGET_METRIC].copy()
    if target_df.empty:
        raise ValueError(f"Target metric '{TARGET_METRIC}' not found in dataset")

    target_df["dt"] = pd.to_datetime(target_df["dt"], utc=True, format="mixed")
    target_df = target_df.loc[target_df["dt"] <= DEVELOPMENT_END].copy()
    target_df = target_df.loc[target_df["value"] != -9999].copy()

    target_df["midday_day"] = np.where(
        target_df["dt"].dt.strftime("%H:%M:%S") <= "12:00:00",
        target_df["dt"].dt.date,
        (target_df["dt"] + pd.Timedelta(days=1)).dt.date,
    )

    daily_target = (
        target_df.groupby("midday_day", as_index=True)["value"]
        .mean()
        .sort_index()
    )
    daily_target.index = pd.to_datetime(daily_target.index)
    return daily_target.astype(float)


def build_summary(daily_target: pd.Series) -> dict[str, float | str | None]:
    mean_value = float(daily_target.mean())
    variance_value = float(daily_target.var(ddof=1))
    std_value = float(daily_target.std(ddof=1))
    zero_rate = float((daily_target == 0).mean())
    fractional_share = float(((daily_target % 1).abs() > 1e-9).mean())
    poisson_zero_rate = float(math.exp(-mean_value))
    max_value = float(daily_target.max())
    q95_value = float(daily_target.quantile(0.95))
    q99_value = float(daily_target.quantile(0.99))
    monthly_mean = daily_target.resample("ME").mean()
    monthly_var = daily_target.resample("ME").var(ddof=1)
    nb_params = fit_negative_binomial(mean_value, variance_value)

    summary: dict[str, float | str | None] = {
        "n_days": int(daily_target.shape[0]),
        "start_date": daily_target.index.min().strftime("%Y-%m-%d"),
        "end_date": daily_target.index.max().strftime("%Y-%m-%d"),
        "mean": mean_value,
        "variance": variance_value,
        "std": std_value,
        "dispersion_index": variance_value / mean_value if mean_value > 0 else None,
        "fractional_share": fractional_share,
        "zero_rate_observed": zero_rate,
        "zero_rate_poisson": poisson_zero_rate,
        "max": max_value,
        "p95": q95_value,
        "p99": q99_value,
        "monthly_mean_average": float(monthly_mean.mean()),
        "monthly_variance_average": float(monthly_var.mean()),
        "monthly_dispersion_average": float((monthly_var / monthly_mean).replace([np.inf, -np.inf], np.nan).dropna().mean()),
    }

    if nb_params is not None:
      size, probability = nb_params
      summary["nb_size"] = size
      summary["nb_probability"] = probability
      summary["zero_rate_negative_binomial"] = float(probability ** size)
    else:
      summary["nb_size"] = None
      summary["nb_probability"] = None
      summary["zero_rate_negative_binomial"] = None

    return summary


def plot_time_series(daily_target: pd.Series, output_dir: Path) -> None:
    rolling_mean = daily_target.rolling(window=28, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_target.index, daily_target.values, color="#5f7c8a", linewidth=1.0, alpha=0.7, label="Daily target")
    ax.plot(rolling_mean.index, rolling_mean.values, color="#c75146", linewidth=2.0, label="28-day rolling mean")
    ax.set_title("Daily Estimated Avoidable Deaths")
    ax.set_ylabel("Deaths")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "target_daily_series.png", dpi=160)
    plt.close(fig)


def plot_distribution(daily_target: pd.Series, output_dir: Path) -> None:
    max_count = int(max(daily_target.max(), 1))
    k_values = np.arange(0, max_count + 1)
    observed = daily_target.round().astype(int).value_counts().sort_index().reindex(k_values, fill_value=0)
    mean_value = float(daily_target.mean())
    poisson_expected = poisson_pmf(k_values, mean_value) * len(daily_target)
    nb_params = fit_negative_binomial(mean_value, float(daily_target.var(ddof=1)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(k_values, observed.values, color="#7aa6c2", alpha=0.8, label="Observed days")
    ax.plot(k_values, poisson_expected, color="#c75146", marker="o", linewidth=2, label="Poisson expected")

    if nb_params is not None:
        size, probability = nb_params
        nb_expected = negative_binomial_pmf(k_values, size, probability) * len(daily_target)
        ax.plot(k_values, nb_expected, color="#2a9d8f", marker="s", linewidth=2, label="NegBin expected")

    ax.set_title("Daily Count Distribution vs Poisson / Negative Binomial")
    ax.set_xlabel("Daily avoidable deaths")
    ax.set_ylabel("Number of days")
    ax.set_xticks(k_values)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "target_count_distribution.png", dpi=160)
    plt.close(fig)


def plot_monthly_panels(daily_target: pd.Series, output_dir: Path) -> None:
    monthly = pd.DataFrame(
        {
            "mean": daily_target.resample("ME").mean(),
            "variance": daily_target.resample("ME").var(ddof=1),
            "zero_rate": daily_target.eq(0).resample("ME").mean(),
        }
    ).dropna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(monthly.index, monthly["mean"], color="#264653", linewidth=2)
    axes[0].set_title("Monthly Mean")
    axes[0].set_ylabel("Mean")
    axes[0].grid(alpha=0.2)

    axes[1].plot(monthly.index, monthly["variance"], color="#e76f51", linewidth=2)
    axes[1].set_title("Monthly Variance")
    axes[1].set_ylabel("Variance")
    axes[1].grid(alpha=0.2)

    axes[2].plot(monthly.index, monthly["zero_rate"], color="#2a9d8f", linewidth=2)
    axes[2].set_title("Monthly Zero Rate")
    axes[2].set_ylabel("Share of zero days")
    axes[2].set_xlabel("Month")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "target_monthly_panels.png", dpi=160)
    plt.close(fig)


def plot_mean_variance(daily_target: pd.Series, output_dir: Path) -> None:
    monthly = pd.DataFrame(
        {
            "mean": daily_target.resample("ME").mean(),
            "variance": daily_target.resample("ME").var(ddof=1),
        }
    ).dropna()

    max_axis = max(monthly["mean"].max(), monthly["variance"].max()) * 1.05

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(monthly["mean"], monthly["variance"], color="#5f0f40", s=55, alpha=0.85)
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#6c757d", label="Poisson line: variance = mean")
    ax.set_title("Monthly Mean-Variance Relationship")
    ax.set_xlabel("Monthly mean")
    ax.set_ylabel("Monthly variance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "target_mean_variance.png", dpi=160)
    plt.close(fig)


def write_summary(summary: dict[str, float | str | None], output_dir: Path) -> None:
    lines = ["Target variable summary", "=======================", ""]

    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")

    interpretation = [
        "",
        "Interpretation hints",
        "--------------------",
    ]

    dispersion = summary.get("dispersion_index")
    observed_zero = summary.get("zero_rate_observed")
    poisson_zero = summary.get("zero_rate_poisson")

    if isinstance(dispersion, float):
        if dispersion > 1.2:
            interpretation.append("Variance exceeds mean materially, which argues against a simple Poisson assumption.")
        elif dispersion < 0.8:
            interpretation.append("Variance is below the mean, so the series is under-dispersed relative to Poisson.")
        else:
            interpretation.append("Variance is close to the mean, so a Poisson baseline is at least plausible.")

    if isinstance(observed_zero, float) and isinstance(poisson_zero, float):
        if observed_zero > poisson_zero + 0.03:
            interpretation.append("Observed zero frequency is above Poisson expectation, suggesting zero inflation or over-dispersion.")
        elif observed_zero < poisson_zero - 0.03:
            interpretation.append("Observed zero frequency is below Poisson expectation, suggesting the mass is shifted toward positive counts.")
        else:
            interpretation.append("Observed zero frequency is close to Poisson expectation.")

    fractional_share = summary.get("fractional_share")
    if isinstance(fractional_share, float) and fractional_share > 0.5:
        interpretation.append("The target is mostly fractional rather than integer-valued, so Poisson or Negative Binomial models should be treated as quasi-likelihood approximations rather than literal count models.")

    (output_dir / "target_summary.txt").write_text("\n".join(lines + interpretation) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the statistical properties of the target variable.")
    parser.add_argument(
        "--zip-path",
        default="data/turingAI_forecasting_challenge_dataset.csv.zip",
        help="Path to the zipped CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="report/target_analysis",
        help="Directory for figures and summary output.",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_target = load_daily_target(zip_path)
    summary = build_summary(daily_target)

    plot_time_series(daily_target, output_dir)
    plot_distribution(daily_target, output_dir)
    plot_monthly_panels(daily_target, output_dir)
    plot_mean_variance(daily_target, output_dir)
    write_summary(summary, output_dir)

    print(f"Saved analysis outputs to {output_dir}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()