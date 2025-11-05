import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def main():
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_csv("data/clean_dataset.csv")
    df.columns = df.columns.str.strip()  # clean column names
    
    target = "Approved"
    X = df.drop(columns=[target])
    y = df[target]

    # ---------------------------
    # EDA
    # ---------------------------
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Class Balance ---")
    print(y.value_counts(normalize=True))
    
    # Numeric features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    
    # Histograms for numeric features
    for col in numeric_features:
        plt.figure(figsize=(5,3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_features + [target]].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
    
    # ---------------------------
    # Preprocessing
    # ---------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # ---------------------------
    # Predictions & Evaluation
    # ---------------------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    # ---------------------------
    # Odds Ratios Interpretation
    # ---------------------------
    model_coefs = clf.named_steps["model"].coef_[0]
    feature_names = numeric_features + list(clf.named_steps["preprocessor"].transformers_[1][1].named_steps["encoder"].get_feature_names_out(categorical_features))
    
    odds_ratios = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model_coefs,
        "Odds_Ratio": np.exp(model_coefs)
    }).sort_values(by="Odds_Ratio", ascending=False)
    
    print("\n--- Odds Ratios ---")
    print(odds_ratios.head(10))
    
    # ---------------------------
    # Responsible Modeling Note
    # ---------------------------
    print("\nNote: Sensitive attributes such as Gender, Ethnicity, and Marital status are included here for demonstration purposes. In a real fintech application, avoid using these features directly to prevent bias in credit approval predictions.")

if __name__ == "__main__":
    main()
