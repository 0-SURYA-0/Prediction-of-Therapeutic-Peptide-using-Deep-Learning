"""
Model Loader for Therapeutic Peptide Prediction
Clean, consistent loader for TensorFlow/Keras models and preprocessors.
"""

import os
import logging
import pickle
from typing import Dict, Any, Optional

import joblib

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and caches models, scalers and metadata from a models directory.
    Names are aligned with the predictor expectations.
    """

    def __init__(self, models_dir: str):
        self.models_dir = os.path.abspath(models_dir)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def load_all_models(self) -> int:
        """Load all known models and preprocessors if present."""
        if not os.path.isdir(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return 0

        loaded = 0
        loaded += self._load_models()
        loaded += self._load_preprocessors()
        loaded += self._load_metadata()
        return loaded

    def _load_models(self) -> int:
        if not HAS_TF:
            logger.warning("TensorFlow not available; skipping .h5 model loading")
            return 0

        model_map = {
            'comprehensive_feature': 'comprehensive_feature_model.h5',
            'feature_extraction': 'feature_extraction_model.h5',
            'multiclass_classifier': 'multiclass_classifier.h5',
            'multiclass_therapeutic_peptide': 'multiclass_therapeutic_peptide.h5',
            'therapeutic_peptide_classifier': 'therapeutic_peptide_classifier.h5',
            'therapeutic_peptide_classifier_15class': 'therapeutic_peptide_classifier_15class.h5',
        }

        count = 0
        for name, filename in model_map.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.isfile(path):
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    logger.info(f"Loaded model: {name} from {filename}")
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load model {name}: {e}")
        return count

    def _load_preprocessors(self) -> int:
        scaler_map = {
            'comprehensive_feature_scaler': 'comprehensive_feature_scaler.pkl',
            'feature_scaler': 'feature_scaler.pkl',
            'multiclass_scaler': 'multiclass_scaler.pkl',
            'therapeutic_scaler_15class': 'therapeutic_scaler_15class.pkl',
            'therapeutic_pca_15class': 'therapeutic_pca_15class.pkl',
            'pca_model': 'pca_model.pkl',
        }

        count = 0
        for name, filename in scaler_map.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.isfile(path):
                try:
                    if filename.endswith('.pkl'):
                        try:
                            self.scalers[name] = joblib.load(path)
                        except Exception:
                            with open(path, 'rb') as f:
                                self.scalers[name] = pickle.load(f)
                    logger.info(f"Loaded preprocessor: {name} from {filename}")
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load preprocessor {name}: {e}")
        return count

    def _load_metadata(self) -> int:
        metadata_map = {
            'therapeutic_classifier_metadata': 'therapeutic_classifier_metadata.pkl',
        }

        count = 0
        for name, filename in metadata_map.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.isfile(path):
                try:
                    with open(path, 'rb') as f:
                        self.metadata[name] = pickle.load(f)
                    logger.info(f"Loaded metadata: {name}")
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load metadata {name}: {e}")
        return count

    def get_model(self, model_name: str) -> Optional[Any]:
        return self.models.get(model_name)

    def get_scaler(self, scaler_name: str) -> Optional[Any]:
        return self.scalers.get(scaler_name)

    def get_metadata(self, metadata_name: str) -> Optional[Any]:
        return self.metadata.get(metadata_name)

    def list_available_models(self) -> list:
        return list(self.models.keys())

    def list_available_scalers(self) -> list:
        return list(self.scalers.keys())

    def get_model_info(self) -> dict:
        info = {
            'models_loaded': len(self.models),
            'preprocessors_loaded': len(self.scalers),
            'metadata_loaded': len(self.metadata),
            'model_details': {}
        }
        for name, model in self.models.items():
            try:
                if hasattr(model, 'input_shape') and hasattr(model, 'output_shape'):
                    info['model_details'][name] = {
                        'type': 'tensorflow',
                        'input_shape': str(getattr(model, 'input_shape', 'unknown')),
                        'output_shape': str(getattr(model, 'output_shape', 'unknown')),
                    }
                else:
                    info['model_details'][name] = {'type': 'unknown'}
            except Exception as e:
                info['model_details'][name] = {'type': 'error', 'error': str(e)}
        return info


def load_models(models_path: Optional[str] = None) -> Dict[str, Any]:
    """Legacy helper to load all artifacts and return a flat dict."""
    if models_path is None:
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    loader = ModelLoader(models_path)
    loader.load_all_models()
    result: Dict[str, Any] = {}
    result.update(loader.models)
    result.update(loader.scalers)
    result.update(loader.metadata)
    return result