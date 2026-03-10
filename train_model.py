import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import networkx as nx
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from routing_engine import RoutingEngine

def main():
    print("=" * 60)
    print("    Placement Predictor & Career Router Training")
    print("=" * 60)
    
    # Configuration
    PLACE_DATA = "collegePlace.csv"
    CAREER_DATA = "Tech_Data_Cleaned.csv"
    OUTPUT_PATH = "placement_artifacts.pkl"
    
    print("[*] Loading placement dataset...")
    df = pd.read_csv(PLACE_DATA)
    
    # Preprocessing
    le_gender = LabelEncoder()
    le_stream = LabelEncoder()
    
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Stream'] = le_stream.fit_transform(df['Stream'])
    
    X = df.drop('PlacedOrNot', axis=1)
    y = df['PlacedOrNot']
    
    feature_names = list(X.columns)
    
    print("\n[*] Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    
    print("\n[*] Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"    [OK] Training complete! Accuracy: {accuracy:.4f}")
    
    print("\n[*] Initializing RoutingEngine from Career Data...")
    routing_engine = RoutingEngine(CAREER_DATA)
    print(f"    [OK] Extracted {len(routing_engine.all_jobs)} jobs and {len(routing_engine.all_unique_skills)} skills.")

    print(f"\n[*] Saving artifacts to '{OUTPUT_PATH}'...")
    artifacts = {
        'model': model,
        'le_gender': le_gender,
        'le_stream': le_stream,
        'feature_names': feature_names,
        'routing_engine': routing_engine
    }
    joblib.dump(artifacts, OUTPUT_PATH)
    print("    [OK] Successfully saved!")
    print("=" * 60)

if __name__ == "__main__":
    main()
