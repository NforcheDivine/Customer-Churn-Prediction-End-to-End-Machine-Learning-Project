import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


# ------------------------------
# 1) Load data
# ------------------------------
df = pd.read_csv("data/Customer_Churn_Joined_Original.csv")

# ------------------------------
# 2) Feature engineering: recency (convert dates -> days since)
# ------------------------------
today = pd.to_datetime("today")

# Convert to datetime safely
for col in ["LastLoginDate", "InteractionDate", "TransactionDate"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "LastLoginDate" in df.columns:
    df["DaysSinceLastLogin"] = (today - df["LastLoginDate"]).dt.days

if "InteractionDate" in df.columns:
    df["DaysSinceLastInteraction"] = (today - df["InteractionDate"]).dt.days

if "TransactionDate" in df.columns:
    df["DaysSinceLastTransaction"] = (today - df["TransactionDate"]).dt.days

# ------------------------------
# 3) Remove leakage columns (raw dates) + IDs
# ------------------------------
date_cols = ["TransactionDate", "InteractionDate", "LastLoginDate"]
df = df.drop(columns=[c for c in date_cols if c in df.columns])

id_cols = ["CustomerID", "TransactionID", "InteractionID"]
df = df.drop(columns=[c for c in id_cols if c in df.columns])

print("Final dataset shape:", df.shape)

# ------------------------------
# 4) Separate features and target
# ------------------------------
X = df.drop("ChurnStatus", axis=1)
y = df["ChurnStatus"]

print("X/y shapes:", X.shape, y.shape)

# ------------------------------
# 5) Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/test shapes:", X_train.shape, X_test.shape)

# ------------------------------
# 6) Identify numeric and categorical columns
# ------------------------------
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------
# 7) Preprocessing pipelines
# ------------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# ------------------------------
# 8) Model
# ------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

# ------------------------------
# 9) Evaluate
# ------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 10) Interpret coefficients
# ------------------------------
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", ascending=False)

print("\nTop 10 positive (increase churn probability):")
print(importance.head(10))

print("\nTop 10 negative (decrease churn probability):")
print(importance.tail(10))
print(confusion_matrix(y_test, y_pred))


# ------------------------------
# SAVE MODEL EVALUATION PLOTS
# ------------------------------
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# 1) ROC Curve
plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Logistic Regression")
plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# 2) Confusion Matrix (visual)
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 3) Feature importance plot (Top 15 by absolute coefficient)
importance_plot = importance.copy()
importance_plot["abs_coef"] = importance_plot["coefficient"].abs()
top = importance_plot.sort_values("abs_coef", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(top["feature"][::-1], top["coefficient"][::-1])
plt.title("Top 15 Features (Logistic Regression Coefficients)")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.savefig(PLOTS_DIR / "feature_importance_top15.png", dpi=150, bbox_inches="tight")
plt.close()

# 4) ROC Curve plot
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

fpr, tpr, _ = roc_curve(y_test, y_prob)

# ADD THIS LINE
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                       name="Logistic Regression")

disp.plot()
plt.title("ROC Curve - Logistic Regression")

plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

from sklearn.metrics import precision_recall_curve, auc

prec, rec, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec, prec)

plt.figure()
plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.savefig(PLOTS_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()



print("\nSaved plots to:", PLOTS_DIR.resolve())
