import json
import pandas as pd
import os

JSON_DIR = "JSON"

def main():
    if not os.path.isdir(JSON_DIR):
        print(f"NO {JSON_DIR}/ folder found")
        return

    rows = []
    for filename in os.listdir(JSON_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(JSON_DIR, filename), "r", encoding="utf-8") as f:
            report = json.load(f)

        td = report.get("technical_details", {})

        if td:
            rows.append(td)

    if not rows:
        print("No reports with technical_details found.")

        return

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} reports. \n")

    for col in ["noise_rms", "contrast", "brightness", "shadow_clip", "highlight_clip"]:
        if col not in df.columns:
            continue
        print(f"-----{col}-----")
        print(df[col].describe()[["mean","std","min","25%","50%","75%","max"]])
        print()


if __name__ == "__main__":
        main()