"""Reproducible cohort selection for Stage 2 pilot.

Uses strictly the training split (train.csv), filters for Approval FY >= 2008
and IPF instrument, and draws a stratified sample of 30 projects using seed 42.
Links official project dates from cleaned_development_program_dataset.csv.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def select_pilot_cohort(
    train_csv: str | Path = "data/processed/train.csv",
    raw_csv: str | Path = "data/raw/cleaned_development_program_dataset.csv",
    output_csv: str | Path = "data/stage2/pilot_cohort.csv",
    random_seed: int = 42,
    target_size: int = 30,
) -> pd.DataFrame:
    train_path = Path(train_csv)
    raw_path = Path(raw_csv)
    out_path = Path(output_csv)

    train = pd.read_csv(train_path)
    raw = pd.read_csv(raw_path)

    # Clean deduplicated raw evaluation dates to match Project ID
    raw_dates = (
        raw.sort_values("IEG_EvalDate", ascending=False)
        .drop_duplicates(subset=["Project ID"], keep="first")[
            ["Project ID", "Approval Date", "Deactivation Date", "IEG_EvalDate"]
        ]
    )

    eligible = train[
        (train["Approval FY"] >= 2008) & (train["Lending Instrument Type"] == "IPF")
    ].copy()

    # Stratified sample proportional to outcome_class
    def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
        n = int(round(target_size * len(g) / len(eligible)))
        return g.sample(n=n, random_state=random_seed)

    sample = (
        eligible.groupby("outcome_class", group_keys=False)
        .apply(_sample_group)
    )

    if len(sample) < target_size:
        rem = target_size - len(sample)
        extra = eligible[~eligible["Project ID"].isin(sample["Project ID"])].sample(
            n=rem, random_state=random_seed
        )
        sample = pd.concat([sample, extra])
    elif len(sample) > target_size:
        sample = sample.sample(n=target_size, random_state=random_seed)

    # Merge official dates
    sample = sample.merge(raw_dates, on="Project ID", how="left")
    sample = sample.sort_values(["outcome_class", "Project ID"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return sample


if __name__ == "__main__":
    df = select_pilot_cohort()
    print(f"Selected {len(df)} projects. Outcome distribution:")
    print(df["outcome_class"].value_counts())
