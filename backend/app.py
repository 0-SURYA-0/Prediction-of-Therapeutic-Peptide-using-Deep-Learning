from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from model_loader import ModelLoader
from predictor import TherapeuticPeptidePredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
model_loader = None
predictor = None

def initialize_models():
    """Initialize model loader and predictor"""
    global model_loader, predictor
    
    try:
        logger.info("Initializing Model Loader...")
        models_path = os.path.join(os.path.dirname(__file__), 'models')
        model_loader = ModelLoader(models_path)
        
        logger.info("Loading models...")
        success = model_loader.load_all_models()
        
        if success:
            logger.info("Initializing Therapeutic Peptide Predictor...")
            predictor = TherapeuticPeptidePredictor(model_loader)
            logger.info("Backend initialization complete!")
            return True
        else:
            logger.error("Failed to load models")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_loader, predictor
    
    status = {
        "status": "healthy" if (model_loader and predictor) else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model_loader.list_available_models() if model_loader else [],
        "scalers_loaded": model_loader.list_available_scalers() if model_loader else []
    }
    
    return jsonify(status)

@app.route('/api/predict/comprehensive', methods=['POST'])
def predict_comprehensive():
    """Comprehensive prediction using all models"""
    global predictor
    
    if not predictor:
        return jsonify({"error": "Predictor not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({"error": "Sequence is required"}), 400
        
        sequence = data['sequence'].strip().upper()
        binary_threshold = data.get('binary_threshold', 0.5)
        multiclass_threshold = data.get('multiclass_threshold', 0.7)
        
        if not predictor.validate_sequence(sequence):
            return jsonify({"error": "Invalid peptide sequence"}), 400
        
        result = predictor.predict_comprehensive(sequence, binary_threshold, multiclass_threshold)
        
        # Add processing timestamp
        result['processed_at'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_sequence():
    """Validate peptide sequence"""
    global predictor
    
    if not predictor:
        return jsonify({"error": "Predictor not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({"error": "Sequence is required"}), 400
        
        sequence = data['sequence'].strip().upper()
        is_valid = predictor.validate_sequence(sequence)
        
        response = {
            "sequence": sequence,
            "is_valid": is_valid,
            "length": len(sequence),
            "validation_rules": {
                "min_length": 3,
                "max_length": 200,
                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error validating sequence: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Legacy endpoint for backward compatibility
@app.route("/predict", methods=["POST"])
def legacy_predict():
    """Legacy prediction endpoint"""
    return predict_comprehensive()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Professional Therapeutic Peptide Prediction API",
        "version": "2.0",
        "endpoints": {
            "/api/health": "Health check and model status",
            "/api/predict/comprehensive": "Comprehensive peptide prediction",
            "/api/validate": "Validate peptide sequence",
            "/predict": "Legacy prediction endpoint (deprecated)"
        },
        "models": {
            "binary_classification": "Therapeutic vs Non-therapeutic",
            "multiclass_classification": "15 therapeutic categories"
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🧬 Initializing Professional Therapeutic Peptide Prediction Backend...")
    
    if initialize_models():
        print("✅ Backend ready! All models loaded successfully.")
        print("🚀 Starting Flask server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize backend - check model files")
        sys.exit(1)
    app.run(host="0.0.0.0", port=5000, debug=True)