import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset."""
    return pd.read_csv(file_path)

def build_preprocessor(numerical_features: list) -> ColumnTransformer:
    """Creates standard scaling transformer for continuous features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def prepare_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Splits features and target into train and test sets with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )