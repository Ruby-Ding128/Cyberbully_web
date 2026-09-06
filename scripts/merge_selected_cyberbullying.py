import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CYBER_PATH = DATA_DIR / "cyberbullying_tweets.csv"
TOXIC_PATH = DATA_DIR / "data.csv"
OUTPUT_PATH = DATA_DIR / "combined_selected_categories.csv"

SELECTED_LABELS = ["age", "ethnicity", "gender", "religion"]


def main() -> None:
    with TOXIC_PATH.open("r", encoding="utf-8-sig", newline="") as input_file:
        toxic_reader = csv.DictReader(input_file)
        if toxic_reader.fieldnames is None:
            raise ValueError("data.csv 没有表头")
        original_columns = toxic_reader.fieldnames

        duplicate_columns = set(original_columns).intersection(SELECTED_LABELS)
        if duplicate_columns:
            raise ValueError(f"新增列与原列重名: {sorted(duplicate_columns)}")

        output_columns = [*original_columns, *SELECTED_LABELS]
        toxic_count = 0
        cyber_counts = {label: 0 for label in SELECTED_LABELS}

        with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()

            for row in toxic_reader:
                row.update({label: 0 for label in SELECTED_LABELS})
                writer.writerow(row)
                toxic_count += 1

            with CYBER_PATH.open("r", encoding="utf-8-sig", newline="") as cyber_file:
                for row in csv.DictReader(cyber_file):
                    label = row.get("cyberbullying_type", "")
                    if label not in SELECTED_LABELS:
                        continue

                    output_row = {column: "" for column in original_columns}
                    output_row["comment_text"] = row.get("tweet_text", "")
                    output_row.update(
                        {selected: int(selected == label) for selected in SELECTED_LABELS}
                    )
                    writer.writerow(output_row)
                    cyber_counts[label] += 1

    print(f"Output: {OUTPUT_PATH}")
    print(f"data.csv rows: {toxic_count}")
    for label in SELECTED_LABELS:
        print(f"{label} rows: {cyber_counts[label]}")
    print(f"selected cyberbullying rows: {sum(cyber_counts.values())}")
    print(f"total rows: {toxic_count + sum(cyber_counts.values())}")


if __name__ == "__main__":
    main()
