import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from preprocess import load_data, prepare_data

def evaluate_model(model_path: str, data_path: str, target_col: str):
    print("1. Loading trained model and test data...")
    pipeline = joblib.load(model_path)
    df = load_data(data_path)
    _, X_test, _, y_test = prepare_data(df, target_col=target_col)
    
    print("2. Generating predictions...")
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    roc_score = roc_auc_score(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    print(f"ROC-AUC Score: {roc_score:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")

if __name__ == "__main__":
    evaluate_model("model.joblib", "data/creditcard.csv", "Class")