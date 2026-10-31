# Telco Customer Churn Analysis
# Data Science and Machine Learning Minor Project
# Run: python main.py

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Make folders for the graphs and result files.
os.makedirs("output/graphs", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD AND CLEAN THE DATA
# ---------------------------------------------------------
data_file = r"C:\Users\ruchi\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_file)

print("\n========== DATASET INFORMATION ==========")
print("Number of rows and columns:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values before cleaning:")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nDuplicate rows:", df.duplicated().sum())

# Remove extra spaces, convert TotalCharges to numeric and remove duplicates.
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop_duplicates()

# Convert the target to numbers: No = 0, Yes = 1.
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# CustomerID is an identifier, so it is not used to predict churn.
df = df.drop(columns=["customerID"], errors="ignore")

print("\n========== CLEANING SUMMARY ==========")
print("Rows and columns after cleaning:", df.shape)
print("Missing values after cleaning:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# 2. OBJECTIVE 1: FIND THE OVERALL CHURN RATE
# ---------------------------------------------------------
print("\n========== OBJECTIVE 1: OVERALL CHURN ==========")
churn_counts = df["Churn"].value_counts().sort_index()
churn_rate = df["Churn"].mean() * 100
print("Customers who stayed:", int(churn_counts.get(0, 0)))
print("Customers who churned:", int(churn_counts.get(1, 0)))
print(f"Overall churn rate: {churn_rate:.2f}%")

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Churn")
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.title("Objective 1 - Overall Customer Churn")
plt.xlabel("Customer status")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("output/graphs/01_overall_churn.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. OBJECTIVE 2: COMPARE CHURN BY CONTRACT TYPE
# ---------------------------------------------------------
print("\n========== OBJECTIVE 2: CHURN BY CONTRACT ==========")
contract_rates = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
print(contract_rates.round(2).to_string())

plt.figure(figsize=(7, 4))
sns.barplot(x=contract_rates.index, y=contract_rates.values)
plt.title("Objective 2 - Churn Rate by Contract Type")
plt.xlabel("Contract type")
plt.ylabel("Churn rate (%)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("output/graphs/02_churn_by_contract.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 4. OBJECTIVE 3: CHECK CUSTOMER TENURE
# ---------------------------------------------------------
print("\n========== OBJECTIVE 3: TENURE AND CHURN ==========")
tenure_summary = df.groupby("Churn")["tenure"].agg(["mean", "median"])
tenure_summary.index = ["Stayed", "Churned"]
print(tenure_summary.round(2))

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=24,
             element="step", stat="density", common_norm=False)
plt.title("Objective 3 - Customer Tenure by Churn")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig("output/graphs/03_tenure_by_churn.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 5. OBJECTIVE 4: COMPARE MONTHLY CHARGES
# ---------------------------------------------------------
print("\n========== OBJECTIVE 4: MONTHLY CHARGES ==========")
charge_summary = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "median"])
charge_summary.index = ["Stayed", "Churned"]
print(charge_summary.round(2))

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.title("Objective 4 - Monthly Charges by Churn")
plt.xlabel("Customer status")
plt.tight_layout()
plt.savefig("output/graphs/04_monthly_charges.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. OBJECTIVE 5: PAYMENT METHOD AND INTERNET SERVICE
# ---------------------------------------------------------
print("\n========== OBJECTIVE 5: PAYMENT METHOD AND SERVICE ==========")
payment_rates = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False) * 100
internet_rates = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
print("\nChurn rate by payment method (%):")
print(payment_rates.round(2).to_string())
print("\nChurn rate by internet service (%):")
print(internet_rates.round(2).to_string())

plt.figure(figsize=(8, 4))
sns.barplot(x=payment_rates.values, y=payment_rates.index)
plt.title("Objective 5 - Churn by Payment Method")
plt.xlabel("Churn rate (%)")
plt.ylabel("Payment method")
plt.tight_layout()
plt.savefig("output/graphs/05_churn_by_payment.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
sns.barplot(x=internet_rates.index, y=internet_rates.values)
plt.title("Additional Analysis - Churn by Internet Service")
plt.xlabel("Internet service")
plt.ylabel("Churn rate (%)")
plt.tight_layout()
plt.savefig("output/graphs/06_churn_by_internet.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 7. PREPARE DATA FOR MACHINE LEARNING
# ---------------------------------------------------------
# Feature engineering: average total charge per month of tenure.
df["AverageChargePerTenureMonth"] = (
    df["TotalCharges"] / df["tenure"].replace(0, float("nan"))
)

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

numeric_columns = X.select_dtypes(include="number").columns.tolist()
category_columns = X.select_dtypes(exclude="number").columns.tolist()

# Missing values, scaling and category encoding are learned from training data.
numeric_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
category_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preparation = ColumnTransformer([
    ("numeric", numeric_steps, numeric_columns),
    ("category", category_steps, category_columns)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 8. OBJECTIVE 6: TRAIN AND COMPARE FIVE ML MODELS
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
}

results = []
trained_models = {}

print("\n========== MACHINE LEARNING MODEL RESULTS ==========")
for model_name, algorithm in models.items():
    model_pipeline = Pipeline([
        ("preparation", preparation),
        ("model", algorithm)
    ])
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })
    trained_models[model_name] = model_pipeline

results_table = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print(results_table.round(3).to_string(index=False))
results_table.to_csv("output/model_comparison.csv", index=False)

# Select the best of the four models using ROC-AUC.
best_model_name = results_table.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nBest model by ROC-AUC:", best_model_name)
print("\nDetailed classification report:")
print(classification_report(y_test, best_predictions,
                            target_names=["Stayed", "Churned"], zero_division=0))

plt.figure(figsize=(8, 4))
sns.barplot(data=results_table, x="ROC-AUC", y="Model")
plt.title("Objective 6 - Machine Learning Model Comparison")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("output/graphs/07_model_comparison.png", dpi=150)
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test, best_predictions, display_labels=["Stayed", "Churned"]
)
plt.title("Best Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("output/graphs/08_confusion_matrix.png", dpi=150)
plt.close()

RocCurveDisplay.from_predictions(y_test, best_probabilities)
plt.title("Best Model - ROC Curve")
plt.tight_layout()
plt.savefig("output/graphs/09_roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 9. SAVE THE MAIN FINDINGS
# ---------------------------------------------------------
with open("output/project_summary.txt", "w", encoding="utf-8") as file:
    file.write("TELCO CUSTOMER CHURN PROJECT SUMMARY\n\n")
    file.write(f"Rows after cleaning: {len(df)}\n")
    file.write(f"Overall churn rate: {churn_rate:.2f}%\n")
    file.write(f"Highest churn contract: {contract_rates.index[0]} ({contract_rates.iloc[0]:.2f}%)\n")
    file.write(f"Highest churn payment method: {payment_rates.index[0]} ({payment_rates.iloc[0]:.2f}%)\n")
    file.write(f"Best model by ROC-AUC: {best_model_name}\n\n")
    file.write(results_table.round(4).to_string(index=False))

print("\nAll done!")
print("Open the 'output' folder to see all graphs and the model comparison CSV.")
