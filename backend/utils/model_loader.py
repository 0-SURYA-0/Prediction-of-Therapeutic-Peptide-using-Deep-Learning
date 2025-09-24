import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import joblib

class ModelLoader:
    def __init__(self, models_dir="../notebook/backend/models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models, scalers, and metadata"""
        try:
            print("🔄 Loading therapeutic peptide models...")
            
            # Load binary classification model (feature-based)
            self._load_feature_model()
            
            # Load multiclass classification models
            self._load_multiclass_models()
            
            # Load ensemble models if available
            self._load_ensemble_models()
            
            print(f"✅ Successfully loaded {len(self.models)} models and {len(self.scalers)} scalers")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def _load_feature_model(self):
        """Load comprehensive feature-based binary classification model"""
        try:
            # Load binary classification model
            model_path = os.path.join(self.models_dir, "comprehensive_feature_model.h5")
            if os.path.exists(model_path):
                self.models['binary_feature'] = load_model(model_path)
                print("  ✓ Binary feature model loaded")
            
            # Load feature scaler
            scaler_path = os.path.join(self.models_dir, "comprehensive_feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers['feature_scaler'] = pickle.load(f)
                print("  ✓ Feature scaler loaded")
                
        except Exception as e:
            print(f"  ⚠️ Error loading feature model: {e}")
    
    def _load_multiclass_models(self):
        """Load multiclass therapeutic category classification models"""
        try:
            # Load main multiclass model
            model_path = os.path.join(self.models_dir, "therapeutic_peptide_classifier_15class.h5")
            if os.path.exists(model_path):
                self.models['multiclass_therapeutic'] = load_model(model_path)
                print("  ✓ 15-class therapeutic classifier loaded")
            
            # Load alternative multiclass model
            alt_model_path = os.path.join(self.models_dir, "multiclass_therapeutic_peptide.h5")
            if os.path.exists(alt_model_path):
                self.models['multiclass_cnn_lstm'] = load_model(alt_model_path)
                print("  ✓ CNN-LSTM multiclass model loaded")
            
            # Load base multiclass classifier
            base_model_path = os.path.join(self.models_dir, "multiclass_classifier.h5")
            if os.path.exists(base_model_path):
                self.models['multiclass_base'] = load_model(base_model_path)
                print("  ✓ Base multiclass classifier loaded")
            
            # Load therapeutic scaler
            scaler_path = os.path.join(self.models_dir, "therapeutic_scaler_15class.pkl")
            if os.path.exists(scaler_path):
                self.scalers['therapeutic_scaler'] = joblib.load(scaler_path)
                print("  ✓ Therapeutic scaler loaded")
            
            # Load multiclass scaler
            mc_scaler_path = os.path.join(self.models_dir, "multiclass_scaler.pkl")
            if os.path.exists(mc_scaler_path):
                with open(mc_scaler_path, 'rb') as f:
                    self.scalers['multiclass_scaler'] = pickle.load(f)
                print("  ✓ Multiclass scaler loaded")
            
            # Load PCA models
            pca_path = os.path.join(self.models_dir, "therapeutic_pca_15class.pkl")
            if os.path.exists(pca_path):
                self.scalers['therapeutic_pca'] = joblib.load(pca_path)
                print("  ✓ Therapeutic PCA loaded")
            
            pca_base_path = os.path.join(self.models_dir, "pca_model.pkl")
            if os.path.exists(pca_base_path):
                self.scalers['pca_base'] = joblib.load(pca_base_path)
                print("  ✓ Base PCA model loaded")
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, "therapeutic_classifier_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata['therapeutic'] = pickle.load(f)
                print("  ✓ Therapeutic metadata loaded")
                
        except Exception as e:
            print(f"  ⚠️ Error loading multiclass models: {e}")
    
    def _load_ensemble_models(self):
        """Load ensemble models if available"""
        try:
            ensemble_files = [f for f in os.listdir(self.models_dir) if 'ensemble' in f.lower()]
            for file in ensemble_files:
                model_name = file.replace('.pkl', '').replace('.h5', '')
                file_path = os.path.join(self.models_dir, file)
                
                if file.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                elif file.endswith('.h5'):
                    self.models[model_name] = load_model(file_path)
                
                print(f"  ✓ {model_name} loaded")
                
        except Exception as e:
            print(f"  ⚠️ Error loading ensemble models: {e}")
    
    def get_model(self, model_name):
        """Get a specific model"""
        return self.models.get(model_name, None)
    
    def get_scaler(self, scaler_name):
        """Get a specific scaler/preprocessor"""
        return self.scalers.get(scaler_name, None)
    
    def get_metadata(self, metadata_name):
        """Get model metadata"""
        return self.metadata.get(metadata_name, None)
    
    def list_available_models(self):
        """List all loaded models"""
        return list(self.models.keys())
    
    def list_available_scalers(self):
        """List all loaded scalers"""
        return list(self.scalers.keys())
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'models': self.list_available_models(),
            'scalers': self.list_available_scalers(),
            'metadata': list(self.metadata.keys()),
            'model_details': {}
        }
        
        for name, model in self.models.items():
            if hasattr(model, 'summary'):
                # TensorFlow/Keras model
                info['model_details'][name] = {
                    'type': 'neural_network',
                    'framework': 'tensorflow',
                    'trainable_params': model.count_params() if hasattr(model, 'count_params') else 'unknown'
                }
            else:
                # Scikit-learn or other model
                info['model_details'][name] = {
                    'type': 'sklearn_model',
                    'class': type(model).__name__
                }
        
        return info
