# LLOYDS Banking Group Customer Churn Prediction – End-to-End Machine Learning Project

## 🧠 Business Context

Customer retention is a strategic priority in retail banking. For organisations such as Lloyds Banking Group, losing a customer impacts long-term revenue, cross-selling opportunities, and brand loyalty. Acquiring new customers is significantly more expensive than retaining existing ones; therefore, the ability to identify customers at risk of churn enables proactive, data-driven intervention.

This project simulates a real analytics engagement aimed at:

- Predicting which customers are likely to leave
- Understanding behavioural drivers of churn
- Supporting targeted retention campaigns
- Demonstrating production-style machine learning engineering

> **Note:** This project uses representative, simulated data and is not affiliated with Lloyds Banking Group.

---

## 🎯 Project Objective

Develop a leakage-free, interpretable machine learning pipeline that:

- Estimates the probability of customer churn
- Highlights key risk indicators
- Provides a reproducible end-to-end workflow
- Aligns with real-world banking analytics practices

---

## 📦 Tools & Technologies

- **Python:** pandas, numpy
- **Machine Learning:** scikit-learn (pipelines, preprocessing, logistic regression)
- **Visualisation:** matplotlib
- **Reporting:** reportlab
- **Environment:** PyCharm, Git, GitHub

---

## 📁 Dataset Overview

The dataset integrates information across multiple domains:

**Demographics**
- Age, gender, marital status, income level

**Engagement & Behaviour**
- Login frequency
- Service usage channel

**Customer Service**
- Interaction type (complaint/feedback)
- Resolution status

**Transactions**
- Amount spent
- Product category

**Temporal Activity**
- Last login
- Last interaction
- Last transaction

**Target Variable**

`ChurnStatus`
- 1 → Customer churned
- 0 → Customer retained

### Key Data Challenges

- Class imbalance (~20% churn)
- Raw date fields causing leakage risk
- Mixed numeric and categorical attributes
- Missing values requiring robust preprocessing

---

## 🔍 Exploratory Data Analysis

EDA focused on generating business insight rather than only technical checks:

- Distribution of engagement metrics
- Churn rate across demographics
- Impact of inactivity on attrition
- Missing value patterns
- Outlier detection

All generated visuals are stored in the `/plots` directory.

---

## ⚙️ Feature Engineering & Leakage Prevention

Initial analysis identified that raw date fields introduced **data leakage** by encoding future information. These were removed and replaced with valid recency features:

- `DaysSinceLastLogin`
- `DaysSinceLastInteraction`
- `DaysSinceLastTransaction`

Identifier columns were excluded to prevent spurious correlations.

### Preprocessing Pipeline

A scikit-learn `ColumnTransformer` ensures reproducibility:

**Numeric Features**
- Median imputation
- Standard scaling

**Categorical Features**
- Most-frequent imputation
- One-hot encoding with unknown handling

---

## 🔢 Modeling Approach

**Baseline Model:** Logistic Regression
**Rationale**

- High interpretability for banking decisions
- Transparent coefficients
- Suitable for limited behavioural history
- Industry-aligned baseline standard

**Imbalance Handling**
- `class_weight="balanced"`
- Evaluation focused on ROC-AUC and recall

---

## 📊 Evaluation Results

**Test Performance**

- ROC-AUC ≈ **0.60**
- Recall ≈ **0.52**
- Precision ≈ **0.24**

### Key Drivers of Churn

**Higher Risk**
- Long time since last interaction
- Low login frequency
- Unresolved complaints
- Low income segment

**Lower Risk**
- Frequent logins
- Resolved service issues
- Website engagement

These align with established banking retention theory, validating model credibility despite moderate baseline performance.

---

## 🧾 Lessons Learned

- Preventing leakage is critical in temporal data
- Business interpretability often outweighs raw accuracy
- Recency is a powerful signal in banking behaviour
- Honest evaluation builds trust with stakeholders

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/eda.py
python src/modeling.py
