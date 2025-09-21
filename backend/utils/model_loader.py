import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_models():
    """
    Load all prediction models and their dependencies.
    
    Returns:
        dict: Dictionary containing all loaded models and their associated transformers.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    models = {}

    try:
        # === Model 1: Therapeutic Prediction ===
        models['model 1'] = load_model(os.path.join(base_dir, 'model 1.h5'))
        models['scaler 1'] = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
        print("✅ Model 1 loaded successfully.")

        # === Model 2: Category Prediction ===
        models['model 2'] = load_model(os.path.join(base_dir, 'model 2.h5'))
        models['scaler 2'] = joblib.load(os.path.join(base_dir, 'scaler 2.pkl'))
        models['pca_model'] = joblib.load(os.path.join(base_dir, 'pca_model.pkl'))
        models['label_encoder'] = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
        models['category_mapping'] = joblib.load(os.path.join(base_dir, 'category_mapping.pkl'))
        print("✅ Model 2 loaded successfully.")

        # === Model 3: Biological Property Prediction ===
        models['model 3'] = load_model(os.path.join(base_dir, 'model 3.h5'))
        models['scaler 3'] = joblib.load(os.path.join(base_dir, 'scaler 3.pkl'))
        print("✅ Model 3 loaded successfully.")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise Exception(f"Failed to load models: {str(e)}")
    
    print("🎉 All models and dependencies loaded successfully.")
    return models
