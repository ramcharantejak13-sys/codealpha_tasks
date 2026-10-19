from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "advertising.csv"
OUT_DIR = ROOT / "outputs"

def prepare_data(df):
    # Case-insensitive mapping for standard Advertising dataset columns.
    lookup = {str(c).strip().lower(): c for c in df.columns}
    required = ["tv", "radio", "newspaper", "sales"]
    missing = [c for c in required if c not in lookup]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. Found columns: {list(df.columns)}"
        )
    clean = df[[lookup[c] for c in required]].copy()
    clean.columns = ["TV", "Radio", "Newspaper", "Sales"]
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna()
    return clean

def train_and_evaluate(df, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df) < 10:
        raise ValueError("At least 10 valid rows are recommended for a train/test split.")
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    results = X_test.copy()
    results["Actual_Sales"] = y_test.to_numpy()
    results["Predicted_Sales"] = pred
    results.to_csv(out_dir / "predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.8)
    low = min(float(y_test.min()), float(pred.min()))
    high = max(float(y_test.max()), float(pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    metrics = [
        f"Test rows: {len(y_test)}",
        f"Mean Absolute Error (MAE): {mae:.4f}",
        f"Root Mean Squared Error (RMSE): {rmse:.4f}",
        f"R-squared (R2): {r2:.4f}",
    ]
    (out_dir / "metrics.txt").write_text("\n".join(metrics) + "\n", encoding="utf-8")
    return model, metrics

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Advertising CSV and save it as data/advertising.csv."
        )
    raw = pd.read_csv(DATA_PATH)
    df = prepare_data(raw)
    _, metrics = train_and_evaluate(df)
    print("\n".join(metrics))
    print(f"\nOutputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
