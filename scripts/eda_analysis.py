import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "combined_selected_categories.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda_analysis"
OUTPUT = OUTPUT_DIR / "eda_metrics.json"

BINARY_LABELS = ["age", "ethnicity", "gender", "religion"]
TOXIC_LABELS = [
    "target",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]


def safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def frame_records(frame: pd.DataFrame):
    return [{str(k): safe(v) for k, v in row.items()} for row in frame.to_dict("records")]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    df["source"] = np.where(df["id"].notna(), "data.csv", "cyberbullying_tweets.csv")
    text = df["comment_text"].fillna("").astype(str)
    df["char_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len()

    source_counts = df["source"].value_counts().rename_axis("source").reset_index(name="count")
    source_counts["share"] = source_counts["count"] / len(df)

    label_rows = []
    for label in BINARY_LABELS:
        count = int((df[label] == 1).sum())
        label_rows.append({"label": label, "count": count, "share_total": count / len(df)})
    binary_summary = pd.DataFrame(label_rows).sort_values("count", ascending=False)

    data_df = df[df["source"] == "data.csv"].copy()
    toxic_rows = []
    for label in TOXIC_LABELS:
        s = pd.to_numeric(data_df[label], errors="coerce")
        toxic_rows.append(
            {
                "label": label,
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "positive_count": int((s > 0).sum()),
                "ge_0_5_count": int((s >= 0.5).sum()),
                "mean": safe(s.mean()),
                "median": safe(s.median()),
                "std": safe(s.std()),
                "min": safe(s.min()),
                "q25": safe(s.quantile(0.25)),
                "q75": safe(s.quantile(0.75)),
                "max": safe(s.max()),
            }
        )
    toxic_summary = pd.DataFrame(toxic_rows)

    length_rows = []
    groups = [("全部数据", df), ("data.csv", data_df), ("cyberbullying_tweets.csv", df[df["source"] != "data.csv"])]
    for name, group in groups:
        for metric, label in [("char_length", "字符数"), ("word_count", "词数")]:
            s = group[metric]
            length_rows.append(
                {
                    "group": name,
                    "metric": label,
                    "count": int(s.count()),
                    "mean": safe(s.mean()),
                    "median": safe(s.median()),
                    "q25": safe(s.quantile(0.25)),
                    "q75": safe(s.quantile(0.75)),
                    "p95": safe(s.quantile(0.95)),
                    "max": safe(s.max()),
                }
            )
    length_summary = pd.DataFrame(length_rows)

    length_bins = [-1, 20, 50, 100, 200, 500, np.inf]
    length_names = ["0–20", "21–50", "51–100", "101–200", "201–500", ">500"]
    length_distribution = (
        pd.cut(df["char_length"], bins=length_bins, labels=length_names)
        .value_counts(sort=False)
        .rename_axis("length_range")
        .reset_index(name="count")
    )
    length_distribution["share"] = length_distribution["count"] / len(df)

    missing_rows = []
    for source_name, group in [("data.csv", data_df), ("cyberbullying_tweets.csv", df[df["source"] != "data.csv"])]:
        for column in df.columns[:49]:
            count = int(group[column].isna().sum())
            missing_rows.append(
                {
                    "source": source_name,
                    "column": column,
                    "missing_count": count,
                    "missing_rate": count / len(group),
                }
            )
    missing_summary = pd.DataFrame(missing_rows).sort_values(
        ["source", "missing_rate", "column"], ascending=[True, False, True]
    )

    corr = data_df[TOXIC_LABELS].apply(pd.to_numeric, errors="coerce").corr()
    corr_table = corr.reset_index().rename(columns={"index": "label"})

    duplicate_count = int(text.duplicated().sum())
    exact_duplicate_groups = int(text[text.duplicated(keep=False)].nunique())
    blank_text_count = int((text.str.strip() == "").sum())

    multi_label_counts = df[BINARY_LABELS].sum(axis=1)
    binary_overlap = (
        multi_label_counts.value_counts().sort_index().rename_axis("active_label_count").reset_index(name="rows")
    )

    samples = []
    for label in BINARY_LABELS:
        sample = df.loc[df[label] == 1, ["comment_text", *BINARY_LABELS]].head(3).copy()
        sample.insert(0, "label", label)
        samples.extend(frame_records(sample))

    metrics = {
        "overview": {
            "rows": int(len(df)),
            "columns": 49,
            "blank_text_count": blank_text_count,
            "duplicate_row_text_count": duplicate_count,
            "duplicate_text_groups": exact_duplicate_groups,
            "selected_cyber_rows": int((df[BINARY_LABELS].sum(axis=1) == 1).sum()),
            "data_rows": int(len(data_df)),
        },
        "source_counts": frame_records(source_counts),
        "binary_summary": frame_records(binary_summary),
        "toxic_summary": frame_records(toxic_summary),
        "length_summary": frame_records(length_summary),
        "length_distribution": frame_records(length_distribution),
        "missing_summary": frame_records(missing_summary),
        "correlation": frame_records(corr_table),
        "binary_overlap": frame_records(binary_overlap),
        "samples": samples,
    }
    OUTPUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(metrics["overview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
