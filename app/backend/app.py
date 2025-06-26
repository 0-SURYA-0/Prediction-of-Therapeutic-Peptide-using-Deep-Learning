from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.predictor import PeptidePredictor
from utils.model_loader import load_models
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access this backend

# Load models using the model_loader function
try:
    print("🔄 Loading models...")
    models_dict = load_models()
    print("✅ All models loaded successfully.")
    
    # Initialize predictor
    predictor = PeptidePredictor(models_dict)
    print("✅ Predictor initialized successfully.")
except Exception as e:
    print(f"❌ Error during initialization: {str(e)}")
    raise e

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict peptide properties from a sequence"""
    data = request.get_json()
    sequence = data.get("sequence", "").strip()

    if not sequence:
        return jsonify({"error": "No sequence provided"}), 400
    
    try:
        result = predictor.predict(sequence)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return jsonify({"status": "API is running"}), 200

# Add a basic route for the root endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Peptide Prediction API",
        "endpoints": {
            "/predict": "POST - Submit a peptide sequence for prediction",
            "/health": "GET - Check if the API is running"
        }
    })

# ✅ REQUIRED to run the app when calling `python app.py`
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)