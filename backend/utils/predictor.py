import numpy as np
import tensorflow as tf
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import BertTokenizer, BertModel, TFBertModel
import os
import joblib
import pickle

class PeptidePredictor:
    def __init__(self, models_dict):
        """
        Initialize the PeptidePredictor with all required models and dependencies

        Args:
            models_dict: Dictionary containing all loaded models and their preprocessors
        """
        self.model_1 = models_dict['model 1']
        self.scaler_1 = models_dict['scaler 1']

        self.model_2 = models_dict['model 2']
        self.scaler_2 = models_dict['scaler 2']
        self.pca_model = models_dict['pca_model']
        self.label_encoder = models_dict['label_encoder']
        self.category_mapping = models_dict['category_mapping']

        self.model_3 = models_dict['model 3']
        self.scaler_3 = models_dict['scaler 3']

        # Initialize ProtBERT for models 1 & 2
        print("Loading ProtBERT models...")
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.tf_bert_model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)
        print("ProtBERT models loaded")

        # Feature names for Model 3
        self.feature_names = [
            "Molecular Weight",
            "Aromaticity",
            "Instability Index",
            "Isoelectric Point",
            "Hydrophobicity (GRAVY)"
        ]

    def validate_sequence(self, sequence):
        """Validate if sequence contains only valid amino acids"""
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not sequence or any(aa not in valid_amino_acids for aa in sequence):
            return False
        return True

    def embed_with_protbert(self, sequence):
        """Generate ProtBERT embeddings for Model 1"""
        # Add spaces between amino acids
        spaced_seq = ' '.join(sequence)
        encoded_input = self.tokenizer(spaced_seq, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs = self.bert_model(**encoded_input)

        # Take mean across all token embeddings (excluding [CLS], [SEP])
        embeddings = outputs.last_hidden_state.squeeze(0)[1:-1].mean(dim=0)
        return embeddings.cpu().numpy().reshape(1, -1)  # shape (1, 1024)

    def preprocess_protbert_tf(self, sequence):
        """Generate ProtBERT embeddings using TensorFlow for Model 2"""
        formatted_seq = " ".join(list(sequence.strip().upper()))
        tokens = self.tokenizer([formatted_seq], return_tensors="tf", padding=True)
        with tf.device("/CPU:0"):  # Use CPU for compatibility
            output = self.tf_bert_model(**tokens)
        embeddings = tf.reduce_mean(output.last_hidden_state, axis=1).numpy()
        return embeddings

    def get_bio_features(self, sequence):
        """Extract biological features for Model 3"""
        try:
            analyzer = ProteinAnalysis(sequence)
            features = [
                analyzer.molecular_weight(),
                analyzer.aromaticity(),
                analyzer.instability_index(),
                analyzer.isoelectric_point(),
                analyzer.gravy()
            ]
            # Format features as dictionary with names
            features_dict = {name: value for name, value in zip(self.feature_names, features)}
            return features, features_dict
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None, None

    def predict_model_1(self, sequence):
        """Run Model 1 to predict if peptide is therapeutic"""
        embedded = self.embed_with_protbert(sequence)
        scaled_input = self.scaler_1.transform(embedded)
        scaled_input = scaled_input.reshape(1, 1024, 1)  # match model input shape

        prediction = self.model_1.predict(scaled_input, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        is_therapeutic = bool(predicted_class == 1)

        return is_therapeutic

    def predict_model_2(self, sequence):
        """Run Model 2 to predict therapeutic categories"""
        embedding = self.preprocess_protbert_tf(sequence)

        # Apply scaler and PCA
        scaled = self.scaler_2.transform(embedding)
        reduced = self.pca_model.transform(scaled)

        # Prediction
        prediction = self.model_2.predict(reduced, verbose=0)

        # Get all categories with their probabilities
        categories = {}

        # Process all classes
        for i, prob in enumerate(prediction[0]):
            # Convert index to label
            label = self.label_encoder.inverse_transform([i])[0]
            # Map to category name
            category = self.category_mapping.get(label, f"Category_{i}")
            # Consider it positive if probability is above threshold
            categories[category] = bool(prob > 0.5)

        return categories

    def predict_model_3(self, sequence):
        """Run Model 3 to predict biological score and extract features"""
        # Extract features
        features, features_dict = self.get_bio_features(sequence)

        if features is None:
            return None

        # Scale and predict
        features_scaled = self.scaler_3.transform([features])
        predicted_score = float(self.model_3.predict(features_scaled, verbose=0)[0][0])

        # Format result as text similar to the example
        result = f"🔡 Enter a peptide sequence (only standard amino acids): {sequence} \n"
        result += "🔬 Extracted Biological Features: \n"

        for name, value in features_dict.items():
            result += f"{name}: {value:.4f} \n"

        result += f"\n Predicted Biological Score: {predicted_score:.4f}"

        return result

    def predict(self, sequence):
        """
        Main prediction pipeline that runs all models in sequence
        Only runs Models 2 and 3 if Model 1 predicts 'Therapeutic'
        """
        # Validate sequence
        sequence = sequence.strip().upper()
        if not self.validate_sequence(sequence):
            return {
                "error": "Invalid sequence. Please enter a valid peptide sequence using only standard amino acids (ACDEFGHIKLMNPQRSTVWY)."
            }

        try:
            # Run Model 1 - Therapeutic prediction
            is_therapeutic = self.predict_model_1(sequence)

            # Initialize result dictionary
            result = {
                "therapeutic": is_therapeutic,
            }

            # Only run Models 2 and 3 if therapeutic
            if is_therapeutic:
                # Run Model 2 - Category prediction
                categories = self.predict_model_2(sequence)
                result["model2_result"] = categories

                # Run Model 3 - Biological properties
                bio_properties = self.predict_model_3(sequence)
                result["model3_result"] = bio_properties

            return result

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}