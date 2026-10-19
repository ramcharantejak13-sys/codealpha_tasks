from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "unemployment.csv"
OUT_DIR = ROOT / "outputs"

def normalize_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

def load_unemployment_data(path):
    df = pd.read_csv(path)
    df.columns = [normalize_name(c) for c in df.columns]
    # Normalize common column names in the commonly used India dataset.
    aliases = {
        "estimated_unemployment_rate": "unemployment_rate",
        "estimated_unemployment_rate_percent": "unemployment_rate",
        "estimated_unemployment_rate_": "unemployment_rate",
        "estimated_unemployment_rate_percent_": "unemployment_rate",
        "region": "region",
        "date": "date",
        "area": "area",
    }
    rename = {}
    for col in df.columns:
        if col in aliases:
            rename[col] = aliases[col]
        elif "unemployment" in col and "rate" in col:
            rename[col] = "unemployment_rate"
        elif col == "estimated_employed":
            rename[col] = "estimated_employed"
        elif "labour_participation" in col or "labor_participation" in col:
            rename[col] = "labour_participation_rate"
    df = df.rename(columns=rename)
    if "unemployment_rate" not in df.columns:
        candidates = [c for c in df.columns if "unemployment" in c and "rate" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "unemployment_rate"})
        else:
            raise ValueError(
                "Could not identify unemployment-rate column. Columns found: "
                + ", ".join(df.columns)
            )
    df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if "region" not in df.columns:
        df["region"] = "Unknown"
    df = df.dropna(subset=["unemployment_rate"])
    return df

def analyze(df, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if "date" in df.columns and df["date"].notna().any():
        dated = df.dropna(subset=["date"]).copy()
        monthly = dated.groupby(pd.Grouper(key="date", freq="MS"))["unemployment_rate"].mean().dropna()
        if not monthly.empty:
            plt.figure(figsize=(10, 5))
            monthly.plot(marker="o")
            plt.title("Average Unemployment Rate Over Time")
            plt.xlabel("Date")
            plt.ylabel("Unemployment rate (%)")
            plt.tight_layout()
            plt.savefig(out_dir / "unemployment_over_time.png", dpi=150)
            plt.close()

    by_region = (df.groupby("region", dropna=False)["unemployment_rate"]
                   .mean().sort_values(ascending=False).head(10))
    if not by_region.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=by_region.values, y=by_region.index)
        plt.title("Top 10 Regions by Average Unemployment Rate")
        plt.xlabel("Average unemployment rate (%)")
        plt.ylabel("Region")
        plt.tight_layout()
        plt.savefig(out_dir / "top_regions_average_unemployment.png", dpi=150)
        plt.close()

    cleaned_path = out_dir / "unemployment_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    summary = [
        f"Rows after cleaning: {len(df)}",
        f"Regions represented: {df['region'].nunique()}",
        f"Mean unemployment rate (%): {df['unemployment_rate'].mean():.2f}",
        f"Median unemployment rate (%): {df['unemployment_rate'].median():.2f}",
        f"Minimum unemployment rate (%): {df['unemployment_rate'].min():.2f}",
        f"Maximum unemployment rate (%): {df['unemployment_rate'].max():.2f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return summary

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the unemployment-in-india CSV and save it as data/unemployment.csv."
        )
    df = load_unemployment_data(DATA_PATH)
    summary = analyze(df)
    print("\n".join(summary))
    print(f"\nOutputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
