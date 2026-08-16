"""
Downloads the real Stanford IMDB Large Movie Review Dataset (50,000
labeled reviews: 25k train + 25k test, balanced pos/neg) and converts it
into the same `text,label` CSV schema as the demo dataset, so it drops
in as a straight replacement for data/raw/reviews.csv.

Source: https://ai.stanford.edu/~amaas/data/sentiment/  (Maas et al., 2011)
Size: ~80MB compressed, ~230MB extracted — this is why it's DVC-tracked
rather than committed to git. See README "Using the full IMDB dataset"
for the DVC + Google Drive remote setup.

Usage:
    python src/prepare_imdb.py                  # full 50k reviews
    python src/prepare_imdb.py --sample-size 10000   # balanced subset,
                                                       # e.g. for faster CI runs
"""
import argparse
import csv
import shutil
import tarfile
import urllib.request
from pathlib import Path

from utils import ROOT

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DOWNLOAD_PATH = ROOT / "data" / "raw" / "_aclImdb_v1.tar.gz"
EXTRACT_DIR = ROOT / "data" / "raw" / "_aclImdb"
OUTPUT_CSV = ROOT / "data" / "raw" / "reviews.csv"


def download(force: bool = False):
    if DOWNLOAD_PATH.exists() and not force:
        print(f"Already downloaded: {DOWNLOAD_PATH}")
        return
    print(f"Downloading {IMDB_URL} (~80MB)...")
    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(IMDB_URL, DOWNLOAD_PATH)
    print("Download complete.")


def extract():
    if EXTRACT_DIR.exists():
        print(f"Already extracted: {EXTRACT_DIR}")
        return
    print("Extracting...")
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR.parent, filter="data")
    (EXTRACT_DIR.parent / "aclImdb").rename(EXTRACT_DIR)
    print("Extraction complete.")


def collect_reviews(split: str, label_dir: str, label: str) -> list[tuple[str, str]]:
    folder = EXTRACT_DIR / split / label_dir
    reviews = []
    for txt_file in sorted(folder.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("<br />", " ").replace("\n", " ").strip()
        reviews.append((text, label))
    return reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Balanced subset size (e.g. 10000 = 5000 pos + 5000 neg). "
        "Default: use all 50,000 reviews.",
    )
    parser.add_argument("--keep-raw", action="store_true",
                         help="Keep the downloaded tar.gz and extracted folder")
    args = parser.parse_args()

    download()
    extract()

    print("Reading review files (this walks 50,000 .txt files, takes a minute)...")
    rows = (
        collect_reviews("train", "pos", "positive")
        + collect_reviews("train", "neg", "negative")
        + collect_reviews("test", "pos", "positive")
        + collect_reviews("test", "neg", "negative")
    )

    if args.sample_size:
        pos = [r for r in rows if r[1] == "positive"][: args.sample_size // 2]
        neg = [r for r in rows if r[1] == "negative"][: args.sample_size // 2]
        rows = pos + neg

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reviews to {OUTPUT_CSV}")

    if not args.keep_raw:
        DOWNLOAD_PATH.unlink(missing_ok=True)
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        print("Cleaned up intermediate download/extract files.")

    print(
        "\nNext steps:\n"
        "  dvc add data/raw/reviews.csv\n"
        "  git add data/raw/reviews.csv.dvc data/raw/.gitignore\n"
        "  git commit -m 'Swap demo data for full IMDB dataset'\n"
        "  dvc push\n"
    )


if __name__ == "__main__":
    main()
