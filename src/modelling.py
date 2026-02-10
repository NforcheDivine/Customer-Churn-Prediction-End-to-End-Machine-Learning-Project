import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
