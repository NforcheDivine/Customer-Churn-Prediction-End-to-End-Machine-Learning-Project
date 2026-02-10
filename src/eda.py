# ==============================
# EDA TEMPLATE - CUSTOMER CHURN
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------
# 1. Paths and setup
# ------------------------------
DATA_FILE = "data/Customer_Churn_Joined_Original.csv"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 4)

# ------------------------------
# 2. Load dataset
# ------------------------------
df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)

# ------------------------------
# 3. Preview data
# ------------------------------
print("\nFirst 10 rows:")
print(df.head(10))

print("\nData info:")
print(df.info())

# ------------------------------
# 4. Target variable check
# ------------------------------
print("\nChurn distribution:")
print(df["ChurnStatus"].value_counts())

print("\nChurn rate:", df["ChurnStatus"].mean())

# Plot churn distribution
counts = df["ChurnStatus"].value_counts().sort_index()

plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("ChurnStatus")
plt.ylabel("Count")
plt.title("Churn Distribution")
plt.savefig(PLOTS_DIR / "churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------
# 5. Missing values analysis
# ------------------------------
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print("\nMissing values (%):")
print(missing_pct)

top_missing = missing_pct.head(10)

plt.figure(figsize=(10, 4))
plt.bar(top_missing.index, top_missing.values)
plt.xticks(rotation=90)
plt.ylabel("% Missing")
plt.title("Top 10 Columns by Missingness")
plt.savefig(PLOTS_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------
# 6. Identify column types
# ------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

# ------------------------------
# 7. Numeric summary statistics
# ------------------------------
print("\nNumeric summary:")
print(df[numeric_cols].describe().T)

# ------------------------------
# 8. Univariate analysis (numeric)
# ------------------------------
NUMERIC_FEATURES = ["Age", "AmountSpent", "LoginFrequency"]

for col in NUMERIC_FEATURES:
    if col not in df.columns:
        continue

    # Histogram
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / f"hist_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot: {col}")
    plt.savefig(PLOTS_DIR / f"box_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------
# 9. Numeric vs churn
# ------------------------------
for col in NUMERIC_FEATURES:
    if col not in df.columns:
        continue

    data_no_churn = df[df["ChurnStatus"] == 0][col].dropna()
    data_churn = df[df["ChurnStatus"] == 1][col].dropna()

    plt.figure()
    plt.boxplot([data_no_churn, data_churn], labels=["No churn", "Churn"])
    plt.title(f"{col} by ChurnStatus")
    plt.ylabel(col)
    plt.savefig(PLOTS_DIR / f"{col}_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------
# 10. Categorical vs churn rate
# ------------------------------
CATEGORICAL_FEATURES = [
    "Gender",
    "MaritalStatus",
    "IncomeLevel",
    "ServiceUsage",
    "ProductCategory",
    "InteractionType",
    "ResolutionStatus"
]

for col in CATEGORICAL_FEATURES:
    if col not in df.columns:
        continue

    churn_rate = df.groupby(col)["ChurnStatus"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    plt.bar(churn_rate.index.astype(str), churn_rate.values)
    plt.xticks(rotation=90)
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {col}")
    plt.savefig(PLOTS_DIR / f"churn_rate_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("\nEDA completed successfully.")
print(f"Plots saved to folder: {PLOTS_DIR.resolve()}")
