"""
Therapeutic Peptide Predictor
Comprehensive prediction system integrating all trained models
"""

import logging
from typing import Dict, Any

import numpy as np

from .feature_extractor import PeptideFeatureExtractor

logger = logging.getLogger(__name__)


class TherapeuticPeptidePredictor:
	"""
	Main predictor class that integrates all models for comprehensive peptide analysis.
	Supports binary classification, multi-class categorization, and feature analysis.
	"""

	def __init__(self, model_loader):
		self.model_loader = model_loader
		self.feature_extractor = PeptideFeatureExtractor()

		self.therapeutic_categories = [
			'Antibacterial/Antimicrobial', 'Anticancer', 'Antifungal', 'Anti-inflammatory',
			'Antioxidant', 'Antiparasitic', 'Antiviral', 'Anti-MRSA', 'Chemotactic',
			'Immunomodulatory', 'Ion Channel Inhibitors', 'Neuroprotective',
			'Protease Inhibitors', 'Synergistic', 'Wound Healing'
		]

		metadata = self.model_loader.get_metadata('therapeutic_classifier_metadata')
		if metadata and isinstance(metadata, dict) and 'category_names' in metadata:
			cats = metadata.get('category_names')
			if isinstance(cats, (list, tuple)) and len(cats) > 0:
				self.therapeutic_categories = list(cats)
				logger.info("Loaded category names from metadata")

	def validate_sequence(self, sequence: str) -> bool:
		return self.feature_extractor.validate_sequence(sequence)

	def predict_therapeutic_binary(self, sequence: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
		if not self.validate_sequence(sequence):
			return {"error": "Invalid peptide sequence"}

		try:
			model = self.model_loader.get_model('comprehensive_feature')
			scaler = self.model_loader.get_scaler('comprehensive_feature_scaler')
			if model and scaler:
				return self._predict_with_comprehensive_model(sequence, model, scaler, confidence_threshold)

			model = self.model_loader.get_model('feature_extraction')
			scaler = self.model_loader.get_scaler('feature_scaler')
			if model and scaler:
				return self._predict_with_feature_model(sequence, model, scaler, confidence_threshold)

			return {"error": "No binary classification model available"}
		except Exception as e:
			logger.error(f"Error in binary prediction: {e}")
			return {"error": f"Prediction failed: {e}"}

	def predict_therapeutic_category(self, sequence: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
		if not self.validate_sequence(sequence):
			return {"error": "Invalid peptide sequence"}

		try:
			model = self.model_loader.get_model('therapeutic_peptide_classifier_15class')
			scaler = self.model_loader.get_scaler('therapeutic_scaler_15class')
			pca = self.model_loader.get_scaler('therapeutic_pca_15class')
			if model and scaler and pca:
				return self._predict_multiclass_with_pca(sequence, model, scaler, pca, confidence_threshold)

			model = self.model_loader.get_model('multiclass_therapeutic_peptide')
			scaler = self.model_loader.get_scaler('multiclass_scaler')
			if model and scaler:
				return self._predict_multiclass_general(sequence, model, scaler, confidence_threshold)

			return {"error": "No multiclass model available"}
		except Exception as e:
			logger.error(f"Error in category prediction: {e}")
			return {"error": f"Category prediction failed: {e}"}

	def predict_biological_properties(self, sequence: str) -> Dict[str, Any]:
		if not self.validate_sequence(sequence):
			return {"error": "Invalid peptide sequence"}

		try:
			features_dict = self.feature_extractor.extract_basic_features(sequence)
			result = {
				"sequence": sequence,
				"biological_features": features_dict,
				"feature_analysis": self._analyze_features(features_dict)
			}
			return result
		except Exception as e:
			logger.error(f"Error extracting biological properties: {e}")
			return {"error": f"Feature extraction failed: {e}"}

	def predict_comprehensive(self, sequence: str, binary_threshold: float = 0.5, multiclass_threshold: float = 0.7) -> Dict[str, Any]:
		if not self.validate_sequence(sequence):
			return {"error": "Invalid peptide sequence"}

		sequence = sequence.upper().strip()
		results: Dict[str, Any] = {
			"sequence": sequence,
			"length": len(sequence),
			"predictions": {}
		}

		binary_result = self.predict_therapeutic_binary(sequence, binary_threshold)
		if "error" not in binary_result:
			results["predictions"]["binary_classification"] = binary_result

		is_therapeutic = bool(binary_result.get("is_therapeutic", False))
		if is_therapeutic:
			category_result = self.predict_therapeutic_category(sequence, multiclass_threshold)
			if "error" not in category_result:
				results["predictions"]["category_classification"] = category_result
		else:
			results["predictions"]["category_classification"] = {
				"message": "Skipped category prediction - not predicted as therapeutic"
			}

		bio_result = self.predict_biological_properties(sequence)
		if "error" not in bio_result:
			results["predictions"]["biological_properties"] = bio_result

		results["summary"] = self._generate_summary(results["predictions"])
		return results

	def _predict_with_comprehensive_model(self, sequence: str, model, scaler, threshold: float) -> Dict[str, Any]:
		features = self.feature_extractor.extract_comprehensive_features(sequence)
		features_scaled = scaler.transform(features.reshape(1, -1))
		prediction = model.predict(features_scaled, verbose=0)
		probability = float(prediction[0][0])
		is_therapeutic = probability > threshold
		return {
			"sequence": sequence,
			"is_therapeutic": is_therapeutic,
			"probability": probability,
			"confidence": probability if is_therapeutic else (1 - probability),
			"model_used": "comprehensive_feature",
			"threshold": threshold
		}

	def _predict_with_feature_model(self, sequence: str, model, scaler, threshold: float) -> Dict[str, Any]:
		features = self.feature_extractor.extract_comprehensive_features(sequence)
		features_scaled = scaler.transform(features.reshape(1, -1))
		prediction = model.predict(features_scaled, verbose=0)
		probability = float(prediction[0][0])
		is_therapeutic = probability > threshold
		return {
			"sequence": sequence,
			"is_therapeutic": is_therapeutic,
			"probability": probability,
			"confidence": probability if is_therapeutic else (1 - probability),
			"model_used": "feature_extraction",
			"threshold": threshold
		}

	def _predict_multiclass_with_pca(self, sequence: str, model, scaler, pca, threshold: float) -> Dict[str, Any]:
		features = self.feature_extractor.extract_comprehensive_features(sequence)
		if len(features) < 1024:
			padded = np.zeros(1024, dtype=features.dtype)
			padded[: len(features)] = features
			features = padded
		else:
			features = features[:1024]

		features_scaled = scaler.transform(features.reshape(1, -1))
		features_pca = pca.transform(features_scaled)
		features_final = features_pca.reshape(features_pca.shape[0], features_pca.shape[1], 1)

		prediction = model.predict(features_final, verbose=0)
		probabilities = prediction[0]
		predicted_class = int(np.argmax(probabilities))
		confidence = float(probabilities[predicted_class])

		if predicted_class < len(self.therapeutic_categories):
			category = self.therapeutic_categories[predicted_class]
		else:
			category = f"Unknown_Category_{predicted_class}"

		top_indices = np.argsort(probabilities)[-3:][::-1]
		top_predictions = []
		for idx in top_indices:
			cat_name = self.therapeutic_categories[idx] if idx < len(self.therapeutic_categories) else f"Unknown_{idx}"
			top_predictions.append({"category": cat_name, "probability": float(probabilities[idx])})

		return {
			"sequence": sequence,
			"predicted_category": category,
			"confidence": confidence,
			"meets_threshold": confidence >= threshold,
			"top_predictions": top_predictions,
			"model_used": "therapeutic_peptide_classifier_15class",
			"threshold": threshold
		}

	def _predict_multiclass_general(self, sequence: str, model, scaler, threshold: float) -> Dict[str, Any]:
		features = self.feature_extractor.extract_comprehensive_features(sequence)
		features_scaled = scaler.transform(features.reshape(1, -1))
		prediction = model.predict(features_scaled, verbose=0)
		probabilities = prediction[0]
		predicted_class = int(np.argmax(probabilities))
		confidence = float(probabilities[predicted_class])
		category = self.therapeutic_categories[predicted_class] if predicted_class < len(self.therapeutic_categories) else f"Unknown_Category_{predicted_class}"
		return {
			"sequence": sequence,
			"predicted_category": category,
			"confidence": confidence,
			"meets_threshold": confidence >= threshold,
			"model_used": "multiclass_therapeutic_peptide",
			"threshold": threshold
		}

	def _analyze_features(self, features_dict: Dict[str, float]) -> Dict[str, str]:
		analysis: Dict[str, str] = {}
		mw = float(features_dict.get('Molecular Weight', 0))
		if mw < 1000:
			analysis['molecular_weight'] = "Very small peptide"
		elif mw < 3000:
			analysis['molecular_weight'] = "Small peptide"
		elif mw < 5000:
			analysis['molecular_weight'] = "Medium-sized peptide"
		else:
			analysis['molecular_weight'] = "Large peptide"

		gravy = float(features_dict.get('Hydrophobicity (GRAVY)', 0))
		if gravy > 0.5:
			analysis['hydrophobicity'] = "Highly hydrophobic"
		elif gravy > 0:
			analysis['hydrophobicity'] = "Moderately hydrophobic"
		elif gravy > -0.5:
			analysis['hydrophobicity'] = "Moderately hydrophilic"
		else:
			analysis['hydrophobicity'] = "Highly hydrophilic"

		pi = float(features_dict.get('Isoelectric Point', 7.0))
		if pi > 9:
			analysis['charge'] = "Highly basic"
		elif pi > 7.5:
			analysis['charge'] = "Basic"
		elif pi < 4:
			analysis['charge'] = "Highly acidic"
		elif pi < 6.5:
			analysis['charge'] = "Acidic"
		else:
			analysis['charge'] = "Neutral"

		return analysis

	def _generate_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
		summary: Dict[str, Any] = {"overall_assessment": "Unknown", "confidence_level": "Low", "key_findings": []}
		binary_pred = predictions.get("binary_classification", {})
		if binary_pred.get("is_therapeutic", False):
			summary["overall_assessment"] = "Therapeutic"
			confidence = float(binary_pred.get("confidence", 0))
			if confidence > 0.9:
				summary["confidence_level"] = "Very High"
			elif confidence > 0.8:
				summary["confidence_level"] = "High"
			elif confidence > 0.7:
				summary["confidence_level"] = "Moderate"
			else:
				summary["confidence_level"] = "Low"
			summary["key_findings"].append(f"Predicted as therapeutic with {confidence:.1%} confidence")
		else:
			summary["overall_assessment"] = "Non-therapeutic"
			summary["key_findings"].append("Predicted as non-therapeutic")

		category_pred = predictions.get("category_classification", {})
		if "predicted_category" in category_pred:
			category = category_pred["predicted_category"]
			cat_confidence = float(category_pred.get("confidence", 0))
			summary["key_findings"].append(f"Predicted category: {category} ({cat_confidence:.1%})")
		return summary

	def get_model_info(self) -> Dict[str, Any]:
		return {
			"available_models": self.model_loader.list_available_models(),
			"available_preprocessors": self.model_loader.list_available_scalers(),
			"therapeutic_categories": self.therapeutic_categories,
			"capabilities": {
				"binary_classification": "therapeutic_vs_non_therapeutic",
				"multiclass_classification": f"{len(self.therapeutic_categories)}_categories",
				"feature_extraction": "65+_biochemical_features",
				"biological_analysis": "molecular_properties"
			}
		}

