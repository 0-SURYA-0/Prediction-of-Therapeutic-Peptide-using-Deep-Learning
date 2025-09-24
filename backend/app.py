"""
Therapeutic Peptide Prediction API
Comprehensive Flask backend with REST API endpoints
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.model_loader import ModelLoader
from utils.predictor import TherapeuticPeptidePredictor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

model_loader = None
predictor = None


def initialize_models() -> bool:
    """Initialize models and predictor on startup."""
    global model_loader, predictor
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.isdir(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
        model_loader = ModelLoader(models_dir)
        loaded = model_loader.load_all_models()
        predictor = TherapeuticPeptidePredictor(model_loader)
        logger.info(f"Models and predictor initialized (artifacts loaded: {loaded})")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.error(traceback.format_exc())
        return False


def _frontend_compatible_response(
    sequence: str,
    binary: Dict[str, Any],
    category: Dict[str, Any],
    bio: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adapt internal predictor outputs to the frontend's expected shape.
    - therapeutic: boolean
    - model2_result: {category_name: true, ...} (only positives)
    - model3_result: multiline string with features and a final score line
    """
    therapeutic = bool(binary.get('is_therapeutic', False))

    # Build category flags (positives only)
    model2_result: Dict[str, bool] = {}
    if therapeutic and category and 'predicted_category' in category:
        cat_name = str(category['predicted_category'])
        key = cat_name.replace(' ', '_').replace('/', '_')
        model2_result[key] = True
        if 'top_predictions' in category:
            for item in category['top_predictions']:
                try:
                    if float(item.get('probability', 0)) >= float(category.get('threshold', 0.7)) and item.get('category') != cat_name:
                        k = str(item['category']).replace(' ', '_').replace('/', '_')
                        model2_result[k] = True
                except Exception:
                    continue

    # Build biological properties string and a derived score
    lines = []
    if bio and 'biological_features' in bio:
        for k, v in bio['biological_features'].items():
            try:
                lines.append(f"{k}: {float(v):.4f}")
            except Exception:
                pass

    # Derive a simple biological score: scale binary probability into 0-20
    prob = float(binary.get('probability', 0.0))
    bio_score = max(0.0, min(20.0, prob * 20.0))
    lines.append(f"Predicted Biological Score: {bio_score:.4f}")

    return {
        'sequence': sequence,
        'therapeutic': therapeutic,
        'model2_result': model2_result if therapeutic else {},
        'model3_result': "\n".join(lines) if lines else f"Predicted Biological Score: {bio_score:.4f}",
        'timestamp': datetime.now().isoformat()
    }


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'active',
        'message': 'Therapeutic Peptide Prediction API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': ['/health', '/predict', '/predict/binary', '/predict/category', '/predict/biological']
    })


@app.route('/health', methods=['GET'])
def health():
    global model_loader, predictor
    return jsonify({
        'status': 'healthy' if (model_loader and predictor) else 'unhealthy',
        'models_loaded': len(model_loader.models) if model_loader else 0,
        'preprocessors_loaded': len(model_loader.scalers) if model_loader else 0,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 503

    data = request.get_json(silent=True) or {}
    sequence = (data.get('sequence') or '').strip().upper()
    if not sequence:
        return jsonify({'error': "Missing 'sequence' in request body"}), 400

    if not predictor.validate_sequence(sequence):
        return jsonify({'error': 'Invalid peptide sequence'}), 400

    try:
        binary_threshold = float(data.get('binary_threshold', 0.5))
        multiclass_threshold = float(data.get('multiclass_threshold', 0.7))
    except Exception:
        return jsonify({'error': 'Invalid thresholds'}), 400

    binary_res = predictor.predict_therapeutic_binary(sequence, binary_threshold)
    category_res = {}
    if binary_res.get('is_therapeutic'):
        category_res = predictor.predict_therapeutic_category(sequence, multiclass_threshold)
    bio_res = predictor.predict_biological_properties(sequence)

    response = _frontend_compatible_response(sequence, binary_res, category_res, bio_res)
    return jsonify(response)


@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 503
    data = request.get_json(silent=True) or {}
    sequence = (data.get('sequence') or '').strip().upper()
    threshold = float(data.get('threshold', 0.5))
    if not predictor.validate_sequence(sequence):
        return jsonify({'error': 'Invalid peptide sequence'}), 400
    res = predictor.predict_therapeutic_binary(sequence, threshold)
    return jsonify(res)


@app.route('/predict/category', methods=['POST'])
def predict_category():
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 503
    data = request.get_json(silent=True) or {}
    sequence = (data.get('sequence') or '').strip().upper()
    threshold = float(data.get('threshold', 0.7))
    if not predictor.validate_sequence(sequence):
        return jsonify({'error': 'Invalid peptide sequence'}), 400
    res = predictor.predict_therapeutic_category(sequence, threshold)
    return jsonify(res)


@app.route('/predict/biological', methods=['POST'])
def predict_biological():
    global predictor
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 503
    data = request.get_json(silent=True) or {}
    sequence = (data.get('sequence') or '').strip().upper()
    if not predictor.validate_sequence(sequence):
        return jsonify({'error': 'Invalid peptide sequence'}), 400
    res = predictor.predict_biological_properties(sequence)
    return jsonify(res)


@app.before_first_request
def startup():
    logger.info('Starting up and initializing models...')
    if not initialize_models():
        logger.error('Initialization failed; API will run with limited functionality')


if __name__ == '__main__':
    ok = initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
import loggingfrom datetime import datetimeimport logging

from typing import Dict, Any

import tracebackfrom datetime import datetime

from datetime import datetime

app = Flask(__name__)

# Import our custom classes

from utils.model_loader import ModelLoaderCORS(app)# Add utils to path

from utils.predictor import TherapeuticPeptidePredictor

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Configure logging

logging.basicConfig(# Simple predictor without heavy ML dependencies

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'class SimplePredictor:from model_loader import ModelLoader

)

logger = logging.getLogger(__name__)    def __init__(self):from predictor import TherapeuticPeptidePredictor



# Initialize Flask app        self.category_names = [

app = Flask(__name__)

CORS(app)            'Antibacterial/Antimicrobial', 'Anticancer', 'Antifungal',# Initialize Flask app



# Global variables for models and predictor            'Anti-inflammatory', 'Antioxidant', 'Antiparasitic', 'Antiviral',app = Flask(__name__)

model_loader = None

predictor = None            'Anti-MRSA', 'Chemotactic', 'Immunomodulatory', CORS(app)



def initialize_models():            'Ion Channel Inhibitors', 'Neuroprotective', 'Protease Inhibitors',

    """Initialize models and predictor on startup."""

    global model_loader, predictor            'Synergistic', 'Wound Healing'# Configure logging

    

    try:        ]logging.basicConfig(level=logging.INFO)

        # Get models directory path

        models_dir = os.path.join(os.path.dirname(__file__), 'models')    logger = logging.getLogger(__name__)

        

        if not os.path.exists(models_dir):    def validate_sequence(self, sequence):

            logger.error(f"Models directory not found: {models_dir}")

            return False        if not sequence or not isinstance(sequence, str):# Global variables for models

        

        logger.info("Initializing model loader...")            return Falsemodel_loader = None

        model_loader = ModelLoader(models_dir)

                predictor = None

        # Load all available models

        loaded_count = model_loader.load_all_models()        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')

        logger.info(f"Successfully loaded {loaded_count} models")

                sequence_upper = sequence.upper()def initialize_models():

        # Initialize predictor

        logger.info("Initializing predictor...")            """Initialize model loader and predictor"""

        predictor = TherapeuticPeptidePredictor(model_loader)

                if not all(aa in valid_aas for aa in sequence_upper):    global model_loader, predictor

        logger.info("Backend initialization complete!")

        return True            return False    

        

    except Exception as e:            try:

        logger.error(f"Failed to initialize models: {str(e)}")

        logger.error(traceback.format_exc())        if len(sequence) < 3 or len(sequence) > 200:        logger.info("Initializing Model Loader...")

        return False

            return False        models_path = os.path.join(os.path.dirname(__file__), 'models')

@app.route('/', methods=['GET'])

def root():                model_loader = ModelLoader(models_path)

    """Root endpoint - API status."""

    return jsonify({        return True        

        "status": "active",

        "message": "Therapeutic Peptide Prediction API",            logger.info("Loading models...")

        "version": "1.0.0",

        "timestamp": datetime.now().isoformat(),    def analyze_sequence(self, sequence):        success = model_loader.load_all_models()

        "endpoints": {

            "/": "API status",        seq_len = len(sequence)        

            "/health": "Health check",

            "/predict": "Comprehensive peptide prediction",                if success:

            "/predict/binary": "Binary therapeutic classification",

            "/predict/category": "Therapeutic category prediction",        # Amino acid composition            logger.info("Initializing Therapeutic Peptide Predictor...")

            "/predict/biological": "Biological properties analysis",

            "/models/info": "Model information",        composition = {}            predictor = TherapeuticPeptidePredictor(model_loader)

            "/validate": "Sequence validation"

        }        for aa in 'ACDEFGHIKLMNPQRSTVWY':            logger.info("Backend initialization complete!")

    })

            composition[aa] = sequence.count(aa) / seq_len            return True

@app.route('/health', methods=['GET'])

def health_check():                else:

    """Health check endpoint."""

    global model_loader, predictor        # Simple physicochemical analysis            logger.error("Failed to load models")

    

    health_status = {        hydrophobic = 'AILMFWV'            return False

        "status": "healthy" if (model_loader and predictor) else "unhealthy",

        "timestamp": datetime.now().isoformat(),        polar = 'NQST'            

        "models_loaded": len(model_loader.models) if model_loader else 0,

        "preprocessors_loaded": len(model_loader.scalers) if model_loader else 0,        charged = 'KRDEH'    except Exception as e:

        "predictor_initialized": predictor is not None

    }        aromatic = 'FWY'        logger.error(f"Error initializing models: {str(e)}")

    

    if health_status["status"] == "unhealthy":                return False

        health_status["error"] = "Models not properly initialized"

        return jsonify(health_status), 503        hydrophobic_ratio = sum(sequence.count(aa) for aa in hydrophobic) / seq_len

    

    return jsonify(health_status)        polar_ratio = sum(sequence.count(aa) for aa in polar) / seq_len@app.route('/api/health', methods=['GET'])



@app.route('/predict', methods=['POST'])        charged_ratio = sum(sequence.count(aa) for aa in charged) / seq_lendef health_check():

def predict_comprehensive():

    """        aromatic_ratio = sum(sequence.count(aa) for aa in aromatic) / seq_len    """Health check endpoint"""

    Comprehensive peptide prediction endpoint.

    Returns binary classification, category prediction, and biological properties.            global model_loader, predictor

    """

    global predictor        return {    

    

    if not predictor:            "length": seq_len,    status = {

        return jsonify({"error": "Predictor not initialized"}), 503

                "composition": composition,        "status": "healthy" if (model_loader and predictor) else "unhealthy",

    try:

        # Get request data            "hydrophobic_ratio": hydrophobic_ratio,        "timestamp": datetime.now().isoformat(),

        data = request.get_json()

        if not data or 'sequence' not in data:            "polar_ratio": polar_ratio,        "models_loaded": model_loader.list_available_models() if model_loader else [],

            return jsonify({"error": "Missing 'sequence' in request body"}), 400

                    "charged_ratio": charged_ratio,        "scalers_loaded": model_loader.list_available_scalers() if model_loader else []

        sequence = data['sequence'].strip()

        if not sequence:            "aromatic_ratio": aromatic_ratio    }

            return jsonify({"error": "Empty sequence provided"}), 400

                }    

        # Get optional thresholds

        binary_threshold = data.get('binary_threshold', 0.5)        return jsonify(status)

        multiclass_threshold = data.get('multiclass_threshold', 0.7)

            def predict_therapeutic(self, sequence):

        # Validate thresholds

        if not (0 <= binary_threshold <= 1):        analysis = self.analyze_sequence(sequence)@app.route('/api/predict/comprehensive', methods=['POST'])

            return jsonify({"error": "binary_threshold must be between 0 and 1"}), 400

        if not (0 <= multiclass_threshold <= 1):        def predict_comprehensive():

            return jsonify({"error": "multiclass_threshold must be between 0 and 1"}), 400

                # Simple heuristic scoring    """Comprehensive prediction using all models"""

        # Perform comprehensive prediction

        result = predictor.predict_comprehensive(        score = 0.5  # base score    global predictor

            sequence, 

            binary_threshold=binary_threshold,            

            multiclass_threshold=multiclass_threshold

        )        # Length preference    if not predictor:

        

        # Add API metadata        if 5 <= analysis["length"] <= 50:        return jsonify({"error": "Predictor not initialized"}), 500

        result["api_info"] = {

            "endpoint": "/predict",            score += 0.25    

            "timestamp": datetime.now().isoformat(),

            "version": "1.0.0"        elif analysis["length"] > 100:    try:

        }

                    score -= 0.15        data = request.get_json()

        return jsonify(result)

                        if not data or 'sequence' not in data:

    except Exception as e:

        logger.error(f"Error in comprehensive prediction: {str(e)}")        # Balance considerations            return jsonify({"error": "Sequence is required"}), 400

        logger.error(traceback.format_exc())

        return jsonify({        if 0.2 <= analysis["hydrophobic_ratio"] <= 0.6:        

            "error": "Internal server error",

            "message": str(e)            score += 0.15        sequence = data['sequence'].strip().upper()

        }), 500

                binary_threshold = data.get('binary_threshold', 0.5)

@app.route('/predict/binary', methods=['POST'])

def predict_binary():        if analysis["polar_ratio"] >= 0.1:        multiclass_threshold = data.get('multiclass_threshold', 0.7)

    """Binary therapeutic classification endpoint."""

    global predictor            score += 0.1        

    

    if not predictor:                if not predictor.validate_sequence(sequence):

        return jsonify({"error": "Predictor not initialized"}), 503

            if 0.1 <= analysis["charged_ratio"] <= 0.4:            return jsonify({"error": "Invalid peptide sequence"}), 400

    try:

        data = request.get_json()            score += 0.1        

        if not data or 'sequence' not in data:

            return jsonify({"error": "Missing 'sequence' in request body"}), 400                result = predictor.predict_comprehensive(sequence, binary_threshold, multiclass_threshold)

        

        sequence = data['sequence'].strip()        # Ensure score is between 0 and 1        

        threshold = data.get('threshold', 0.5)

                score = max(0.1, min(0.95, score))        # Add processing timestamp

        if not (0 <= threshold <= 1):

            return jsonify({"error": "threshold must be between 0 and 1"}), 400                result['processed_at'] = datetime.now().isoformat()

        

        result = predictor.predict_therapeutic_binary(sequence, threshold)        # Random category for demo        

        

        # Add API metadata        category = random.choice(self.category_names)        return jsonify(result)

        result["api_info"] = {

            "endpoint": "/predict/binary",                

            "timestamp": datetime.now().isoformat()

        }        return {    except Exception as e:

        

        return jsonify(result)            "sequence": sequence,        logger.error(f"Error in comprehensive prediction: {str(e)}")

        

    except Exception as e:            "is_therapeutic": score > 0.6,        return jsonify({"error": str(e)}), 500

        logger.error(f"Error in binary prediction: {str(e)}")

        return jsonify({            "therapeutic_probability": round(score, 3),

            "error": "Internal server error",

            "message": str(e)            "predicted_category": category,@app.route('/api/validate', methods=['POST'])

        }), 500

            "category_confidence": round(score * 0.85, 3),def validate_sequence():

@app.route('/predict/category', methods=['POST'])

def predict_category():            "sequence_analysis": analysis,    """Validate peptide sequence"""

    """Therapeutic category prediction endpoint."""

    global predictor            "timestamp": datetime.now().isoformat()    global predictor

    

    if not predictor:        }    

        return jsonify({"error": "Predictor not initialized"}), 503

        if not predictor:

    try:

        data = request.get_json()# Initialize predictor        return jsonify({"error": "Predictor not initialized"}), 500

        if not data or 'sequence' not in data:

            return jsonify({"error": "Missing 'sequence' in request body"}), 400predictor = SimplePredictor()    

        

        sequence = data['sequence'].strip()    try:

        threshold = data.get('threshold', 0.7)

        @app.route('/', methods=['GET'])        data = request.get_json()

        if not (0 <= threshold <= 1):

            return jsonify({"error": "threshold must be between 0 and 1"}), 400def home():        if not data or 'sequence' not in data:

        

        result = predictor.predict_therapeutic_category(sequence, threshold)    return jsonify({            return jsonify({"error": "Sequence is required"}), 400

        

        # Add API metadata        "message": "🧬 Therapeutic Peptide Prediction API",        

        result["api_info"] = {

            "endpoint": "/predict/category",        "version": "1.0 (Simplified)",        sequence = data['sequence'].strip().upper()

            "timestamp": datetime.now().isoformat()

        }        "status": "Ready",        is_valid = predictor.validate_sequence(sequence)

        

        return jsonify(result)        "endpoints": {        

        

    except Exception as e:            "/": "API information",        response = {

        logger.error(f"Error in category prediction: {str(e)}")

        return jsonify({            "/health": "Health check",            "sequence": sequence,

            "error": "Internal server error",

            "message": str(e)            "/predict": "Peptide prediction",            "is_valid": is_valid,

        }), 500

            "/validate": "Sequence validation"            "length": len(sequence),

@app.route('/predict/biological', methods=['POST'])

def predict_biological():        }            "validation_rules": {

    """Biological properties analysis endpoint."""

    global predictor    })                "min_length": 3,

    

    if not predictor:                "max_length": 200,

        return jsonify({"error": "Predictor not initialized"}), 503

    @app.route('/health', methods=['GET'])                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"

    try:

        data = request.get_json()def health():            }

        if not data or 'sequence' not in data:

            return jsonify({"error": "Missing 'sequence' in request body"}), 400    return jsonify({        }

        

        sequence = data['sequence'].strip()        "status": "healthy",        

        

        result = predictor.predict_biological_properties(sequence)        "timestamp": datetime.now().isoformat(),        return jsonify(response)

        

        # Add API metadata        "predictor": "active",        

        result["api_info"] = {

            "endpoint": "/predict/biological",        "categories": len(predictor.category_names)    except Exception as e:

            "timestamp": datetime.now().isoformat()

        }    })        logger.error(f"Error validating sequence: {str(e)}")

        

        return jsonify(result)        return jsonify({"error": str(e)}), 500

        

    except Exception as e:@app.route('/predict', methods=['POST'])

        logger.error(f"Error in biological analysis: {str(e)}")

        return jsonify({def predict():# Legacy endpoint for backward compatibility

            "error": "Internal server error",

            "message": str(e)    try:@app.route("/predict", methods=["POST"])

        }), 500

        data = request.get_json()def legacy_predict():

@app.route('/validate', methods=['POST'])

def validate_sequence():        if not data or 'sequence' not in data:    """Legacy prediction endpoint"""

    """Validate peptide sequence format."""

    global predictor            return jsonify({"error": "Sequence is required"}), 400    return predict_comprehensive()

    

    if not predictor:        

        return jsonify({"error": "Predictor not initialized"}), 503

            sequence = data['sequence'].strip().upper()@app.route("/", methods=["GET"])

    try:

        data = request.get_json()        def home():

        if not data or 'sequence' not in data:

            return jsonify({"error": "Missing 'sequence' in request body"}), 400        if not predictor.validate_sequence(sequence):    return jsonify({

        

        sequence = data['sequence'].strip()            return jsonify({        "message": "Professional Therapeutic Peptide Prediction API",

        is_valid = predictor.validate_sequence(sequence)

                        "error": "Invalid peptide sequence",        "version": "2.0",

        result = {

            "sequence": sequence,                "requirements": {        "endpoints": {

            "is_valid": is_valid,

            "length": len(sequence),                    "length": "3-200 amino acids",            "/api/health": "Health check and model status",

            "api_info": {

                "endpoint": "/validate",                    "valid_characters": "ACDEFGHIKLMNPQRSTVWY"            "/api/predict/comprehensive": "Comprehensive peptide prediction",

                "timestamp": datetime.now().isoformat()

            }                }            "/api/validate": "Validate peptide sequence",

        }

                    }), 400            "/predict": "Legacy prediction endpoint (deprecated)"

        if not is_valid:

            result["validation_errors"] = [                },

                "Contains invalid amino acid characters" if any(c not in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence.upper()) else None,

                "Sequence too short (minimum 3 amino acids)" if len(sequence) < 3 else None,        result = predictor.predict_therapeutic(sequence)        "models": {

                "Sequence too long (maximum 1000 amino acids)" if len(sequence) > 1000 else None

            ]        return jsonify(result)            "binary_classification": "Therapeutic vs Non-therapeutic",

            result["validation_errors"] = [e for e in result["validation_errors"] if e is not None]

                            "multiclass_classification": "15 therapeutic categories"

        return jsonify(result)

            except Exception as e:        }

    except Exception as e:

        logger.error(f"Error in sequence validation: {str(e)}")        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500    })

        return jsonify({

            "error": "Internal server error",

            "message": str(e)

        }), 500@app.route('/validate', methods=['POST'])@app.errorhandler(404)



@app.route('/models/info', methods=['GET'])def validate():def not_found(error):

def models_info():

    """Get information about loaded models and capabilities."""    try:    return jsonify({"error": "Endpoint not found"}), 404

    global predictor, model_loader

            data = request.get_json()

    if not predictor or not model_loader:

        return jsonify({"error": "Models not initialized"}), 503        if not data or 'sequence' not in data:@app.errorhandler(500)

    

    try:            return jsonify({"error": "Sequence is required"}), 400def internal_error(error):

        info = predictor.get_model_info()

                    return jsonify({"error": "Internal server error"}), 500

        # Add additional system information

        info["system_info"] = {        sequence = data['sequence'].strip().upper()

            "models_directory": os.path.join(os.path.dirname(__file__), 'models'),

            "total_models": len(model_loader.models),        is_valid = predictor.validate_sequence(sequence)if __name__ == '__main__':

            "total_preprocessors": len(model_loader.scalers),

            "api_version": "1.0.0",            print("🧬 Initializing Professional Therapeutic Peptide Prediction Backend...")

            "timestamp": datetime.now().isoformat()

        }        return jsonify({    

        

        return jsonify(info)            "sequence": sequence,    if initialize_models():

        

    except Exception as e:            "is_valid": is_valid,        print("✅ Backend ready! All models loaded successfully.")

        logger.error(f"Error getting model info: {str(e)}")

        return jsonify({            "length": len(sequence),        print("🚀 Starting Flask server on http://localhost:5000")

            "error": "Internal server error",

            "message": str(e)            "validation_rules": {        app.run(host='0.0.0.0', port=5000, debug=True)

        }), 500

                "min_length": 3,    else:

@app.route('/categories', methods=['GET'])

def get_categories():                "max_length": 200,        print("❌ Failed to initialize backend - check model files")

    """Get list of available therapeutic categories."""

    global predictor                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"        sys.exit(1)

    

    if not predictor:            }    app.run(host="0.0.0.0", port=5000, debug=True)

        return jsonify({"error": "Predictor not initialized"}), 503        })

            

    return jsonify({    except Exception as e:

        "categories": predictor.therapeutic_categories,        return jsonify({"error": str(e)}), 500

        "count": len(predictor.therapeutic_categories),

        "api_info": {@app.errorhandler(404)

            "endpoint": "/categories",def not_found(error):

            "timestamp": datetime.now().isoformat()    return jsonify({"error": "Endpoint not found"}), 404

        }

    })@app.errorhandler(500)

def internal_error(error):

@app.errorhandler(404)    return jsonify({"error": "Internal server error"}), 500

def not_found(error):

    """Handle 404 errors."""if __name__ == '__main__':

    return jsonify({    print("🧬 Starting Therapeutic Peptide Prediction API...")

        "error": "Endpoint not found",    print("✅ Simple predictor loaded")

        "message": "Please check the API documentation for available endpoints",    print("🚀 Server running on http://localhost:5000")

        "available_endpoints": ["/", "/health", "/predict", "/predict/binary",     

                               "/predict/category", "/predict/biological",     app.run(host='0.0.0.0', port=5000, debug=True)
                               "/models/info", "/validate", "/categories"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

@app.before_first_request
def startup():
    """Initialize models before first request."""
    logger.info("Starting up application...")
    success = initialize_models()
    if not success:
        logger.error("Failed to initialize models - some endpoints may not work")

if __name__ == '__main__':
    # Initialize models
    logger.info("Initializing Therapeutic Peptide Prediction API...")
    
    if initialize_models():
        logger.info("Models initialized successfully. Starting Flask server...")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    else:
        logger.error("Failed to initialize models. Exiting...")
        exit(1)