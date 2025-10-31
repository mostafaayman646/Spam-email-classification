import joblib

def save_model(pipe, path="Model/XGBoost_pipeline.pkl"):
    """Save the full pipeline (preprocessing + model)."""
    joblib.dump(pipe, path)
    print(f"Pipeline saved at: {path}")