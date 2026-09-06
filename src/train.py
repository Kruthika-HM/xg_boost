import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from preprocess import load_data, prepare_data, build_preprocessor

def train_pipeline(data_path: str, target_col: str, model_save_path: str = "model.joblib"):
    # 1. Load Data
    print("1. Loading dataset...")
    df = load_data(data_path)
    
    # 2. Split Data
    print("2. Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_data(df, target_col=target_col)
    
    # 3. Identify Numerical Features
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 4. Handle Imbalanced Data Class Weights
    # Fraud dataset has few positive (1) cases; scale_pos_weight balances this ratio
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 5. Build Scikit-Learn Pipeline
    print("3. Assembling preprocessing and XGBoost pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor(num_cols)),
        ('classifier', XGBClassifier(
            scale_pos_weight=ratio,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    # 6. Train Model
    print("4. Training XGBoost model (this may take 10-30 seconds)...")
    pipeline.fit(X_train, y_train)
    
    # 7. Save Artifacts
    joblib.dump(pipeline, model_save_path)
    print(f"SUCCESS: Pipeline model saved directly to {model_save_path}")

if __name__ == "__main__":
    train_pipeline("data/creditcard.csv", target_col="Class")