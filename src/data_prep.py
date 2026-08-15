"""
Stage 1 of the pipeline: load raw labeled data, clean it, split into
train/test, and record baseline statistics (text length distribution,
class balance) that monitor_drift.py will compare future production
traffic against.
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import ROOT, load_params, clean_text


def main():
    params = load_params()["data_prep"]

    raw_path = ROOT / "data" / "raw" / "reviews.csv"
    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["label"],
    )

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Baseline stats for drift monitoring — this is the "ground truth"
    # snapshot of what training data looked like, so we can later tell
    # whether production traffic has drifted away from it.
    lengths = df["text"].str.split().apply(len)
    class_counts = df["label"].value_counts(normalize=True).to_dict()

    baseline_stats = {
        "n_samples": len(df),
        "text_length_mean": float(lengths.mean()),
        "text_length_std": float(lengths.std()),
        "text_lengths_sample": lengths.tolist(),  # used for KS test later
        "class_distribution": class_counts,
        "vocab_size_approx": int(df["text"].str.split().explode().nunique()),
    }
    with open(out_dir / "baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"Prepared {len(train_df)} train / {len(test_df)} test rows.")
    print(f"Baseline stats saved to {out_dir / 'baseline_stats.json'}")


if __name__ == "__main__":
    main()
