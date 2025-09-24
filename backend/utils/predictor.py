import numpy as npimport numpy as npimport numpy as np

from collections import Counter

from Bio.SeqUtils.ProtParam import ProteinAnalysisfrom collections import Counterfrom collections import Counter

from sklearn.preprocessing import StandardScaler

import hashlibfrom Bio.SeqUtils.ProtParam import ProteinAnalysisfrom Bio.SeqUtils.ProtParam import ProteinAnalysis



class TherapeuticPeptidePredictor:from sklearn.preprocessing import StandardScalerfrom sklearn.preprocessing import StandardScaler

    """Professional therapeutic peptide prediction system using comprehensive models"""

    import hashlibimport hashlib

    def __init__(self, model_loader):

        self.model_loader = model_loader

        self.category_names = [

            'Antibacterial/Antimicrobial', 'Anticancer', 'Antifungal',class TherapeuticPeptidePredictor:class TherapeuticPeptidePredictor:

            'Anti-inflammatory', 'Antioxidant', 'Antiparasitic', 'Antiviral',

            'Anti-MRSA', 'Chemotactic', 'Immunomodulatory',     def __init__(self, model_loader):    def __init__(self, model_loader):

            'Ion Channel Inhibitors', 'Neuroprotective', 'Protease Inhibitors',

            'Synergistic', 'Wound Healing'        self.model_loader = model_loader        self.model_loader = model_loader

        ]

                self.category_names = [        self.category_names = [

        # Get metadata if available

        metadata = self.model_loader.get_metadata('therapeutic')            'Antibacterial/Antimicrobial',            'Antibacterial/Antimicrobial',

        if metadata and 'category_names' in metadata:

            self.category_names = metadata['category_names']            'Anticancer',             'Anticancer', 

    

    def extract_comprehensive_features(self, seq):            'Antifungal',            'Antifungal',

        """Extract comprehensive features for therapeutic peptide prediction"""

        features = []            'Anti-inflammatory',            'Anti-inflammatory',

        seq_len = len(seq)

        features.append(seq_len)            'Antioxidant',            'Antioxidant',

        

        # Amino acid composition (20 features)            'Antiparasitic',            'Antiparasitic',

        for aa in 'ACDEFGHIKLMNPQRSTVWY':

            features.append(seq.count(aa) / seq_len)            'Antiviral',            'Antiviral',

        

        # Dipeptide composition (20 features)            'Anti-MRSA',            'Anti-MRSA',

        dipeptides = ['AA', 'AC', 'AG', 'AL', 'AR', 'AS', 'AT', 'AV', 'AY', 'AW',

                      'CA', 'CC', 'CG', 'CL', 'CR', 'CS', 'CT', 'CV', 'CY', 'CW']            'Chemotactic',            'Chemotactic',

        dipeptide_count = {}

        for i in range(len(seq) - 1):            'Immunomodulatory',            'Immunomodulatory',

            dipeptide = seq[i:i+2]

            dipeptide_count[dipeptide] = dipeptide_count.get(dipeptide, 0) + 1            'Ion Channel Inhibitors',            'Ion Channel Inhibitors',

        

        for dp in dipeptides:            'Neuroprotective',            'Neuroprotective',

            features.append(dipeptide_count.get(dp, 0) / (seq_len - 1) if seq_len > 1 else 0)

                    'Protease Inhibitors',            'Protease Inhibitors',

        # Physicochemical properties

        try:            'Synergistic',            'Synergistic',

            analyzed_seq = ProteinAnalysis(seq)

            features.extend([            'Wound Healing'            'Wound Healing'

                analyzed_seq.molecular_weight(),

                analyzed_seq.aromaticity(),        ]        ]

                analyzed_seq.instability_index(),

                analyzed_seq.isoelectric_point(),                

                analyzed_seq.gravy()

            ])        # Get metadata if available        # Get metadata if available

            sec_struct = analyzed_seq.secondary_structure_fraction()

            features.extend([sec_struct[0], sec_struct[1], sec_struct[2]])        metadata = self.model_loader.get_metadata('therapeutic')        metadata = self.model_loader.get_metadata('therapeutic')

        except:

            features.extend([0] * 8)        if metadata and 'category_names' in metadata:        if metadata and 'category_names' in metadata:

        

        # Hydrophobicity groups            self.category_names = metadata['category_names']            self.category_names = metadata['category_names']

        hydrophobic = 'AILMFWV'

        polar = 'NQST'         

        charged = 'KRDEH'

        aromatic = 'FWY'    def extract_comprehensive_features(self, seq):    def extract_comprehensive_features(self, seq):

        

        features.extend([        """Extract comprehensive features for therapeutic peptide prediction"""        """Extract comprehensive features for therapeutic peptide prediction"""

            sum(seq.count(aa) for aa in hydrophobic) / seq_len,

            sum(seq.count(aa) for aa in polar) / seq_len,        features = []        features = []

            sum(seq.count(aa) for aa in charged) / seq_len,

            sum(seq.count(aa) for aa in aromatic) / seq_len,        seq_len = len(seq)        seq_len = len(seq)

        ])

                        

        # Charge properties

        positive_charge = seq.count('K') + seq.count('R') + seq.count('H')        # 1. Basic sequence properties        # 1. Basic sequence properties

        negative_charge = seq.count('D') + seq.count('E')

        net_charge = positive_charge - negative_charge        features.append(seq_len)        features.append(seq_len)

        

        features.extend([                

            positive_charge / seq_len,

            negative_charge / seq_len,        # 2. Amino acid composition (20 features)        # 2. Amino acid composition (20 features)

            net_charge / seq_len,

            abs(net_charge) / seq_len,        aa_composition = {}        aa_composition = {}

        ])

                for aa in 'ACDEFGHIKLMNPQRSTVWY':        for aa in 'ACDEFGHIKLMNPQRSTVWY':

        return np.array(features)

                aa_composition[aa] = seq.count(aa) / seq_len            aa_composition[aa] = seq.count(aa) / seq_len

    def validate_sequence(self, sequence):

        """Validate peptide sequence"""            features.append(aa_composition[aa])            features.append(aa_composition[aa])

        if not sequence or not isinstance(sequence, str):

            return False                

        

        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')        # 3. Dipeptide composition (top 20 most important dipeptides)        # 3. Dipeptide composition (top 20 most important dipeptides)

        sequence_upper = sequence.upper()

                dipeptides = ['AA', 'AC', 'AG', 'AL', 'AR', 'AS', 'AT', 'AV', 'AY', 'AW',        dipeptides = ['AA', 'AC', 'AG', 'AL', 'AR', 'AS', 'AT', 'AV', 'AY', 'AW',

        if not all(aa in valid_aas for aa in sequence_upper):

            return False                      'CA', 'CC', 'CG', 'CL', 'CR', 'CS', 'CT', 'CV', 'CY', 'CW']                      'CA', 'CC', 'CG', 'CL', 'CR', 'CS', 'CT', 'CV', 'CY', 'CW']

        

        if len(sequence) < 3 or len(sequence) > 200:        dipeptide_count = {}        dipeptide_count = {}

            return False

                for i in range(len(seq) - 1):        for i in range(len(seq) - 1):

        return True

                dipeptide = seq[i:i+2]            dipeptide = seq[i:i+2]

    def predict_binary_therapeutic(self, sequence, confidence_threshold=0.5):

        """Predict if peptide is therapeutic (binary classification)"""            dipeptide_count[dipeptide] = dipeptide_count.get(dipeptide, 0) + 1            dipeptide_count[dipeptide] = dipeptide_count.get(dipeptide, 0) + 1

        try:

            model = self.model_loader.get_model('comprehensive_feature')                

            scaler = self.model_loader.get_scaler('comprehensive_feature_scaler')

                    for dp in dipeptides:        for dp in dipeptides:

            if not model or not scaler:

                return {"error": "Binary classification model not available"}            features.append(dipeptide_count.get(dp, 0) / (seq_len - 1) if seq_len > 1 else 0)            features.append(dipeptide_count.get(dp, 0) / (seq_len - 1) if seq_len > 1 else 0)

            

            features = self.extract_comprehensive_features(sequence)                

            features_scaled = scaler.transform(features.reshape(1, -1))

            prediction_prob = model.predict(features_scaled, verbose=0)[0][0]        # 4. Physicochemical properties using BioPython        # 4. Physicochemical properties using BioPython

            is_therapeutic = prediction_prob > confidence_threshold

                    try:        try:

            return {

                "sequence": sequence,            analyzed_seq = ProteinAnalysis(seq)            analyzed_seq = ProteinAnalysis(seq)

                "is_therapeutic": bool(is_therapeutic),

                "confidence": float(prediction_prob),            features.extend([            features.extend([

                "prediction_type": "binary_therapeutic",

                "model_used": "comprehensive_feature"                analyzed_seq.molecular_weight(),                analyzed_seq.molecular_weight(),

            }

        except Exception as e:                analyzed_seq.aromaticity(),                analyzed_seq.aromaticity(),

            return {"error": f"Binary prediction failed: {str(e)}"}

                    analyzed_seq.instability_index(),                analyzed_seq.instability_index(),

    def predict_therapeutic_category(self, sequence, confidence_threshold=0.7):

        """Predict therapeutic category (15-class classification)"""                analyzed_seq.isoelectric_point(),                analyzed_seq.isoelectric_point(),

        try:

            model = self.model_loader.get_model('multiclass_therapeutic_peptide')                analyzed_seq.gravy(),  # Grand average of hydropathicity                analyzed_seq.gravy(),  # Grand average of hydropathicity

            scaler = self.model_loader.get_scaler('multiclass_scaler')

                        ])            ])

            if not model or not scaler:

                return {"error": "Multiclass model not available"}                        

            

            features = self.extract_comprehensive_features(sequence)            # Secondary structure fractions            # Secondary structure fractions

            features_scaled = scaler.transform(features.reshape(1, -1))

            prediction_probs = model.predict(features_scaled, verbose=0)[0]            sec_struct = analyzed_seq.secondary_structure_fraction()            sec_struct = analyzed_seq.secondary_structure_fraction()

            

            predicted_class = np.argmax(prediction_probs)            features.extend([sec_struct[0], sec_struct[1], sec_struct[2]])  # helix, turn, sheet            features.extend([sec_struct[0], sec_struct[1], sec_struct[2]])  # helix, turn, sheet

            confidence = np.max(prediction_probs)

                                    

            category = (self.category_names[predicted_class] 

                       if predicted_class < len(self.category_names)         except Exception as e:        except Exception as e:

                       else f"Unknown_Category_{predicted_class}")

                        # If BioPython fails, add zeros            # If BioPython fails, add zeros

            return {

                "sequence": sequence,            features.extend([0] * 8)            features.extend([0] * 8)

                "predicted_category": category,

                "confidence": float(confidence),                

                "meets_threshold": confidence >= confidence_threshold,

                "prediction_type": "multiclass_therapeutic",        # 5. Hydrophobicity groups        # 5. Hydrophobicity groups

            }

        except Exception as e:        hydrophobic = 'AILMFWV'        hydrophobic = 'AILMFWV'

            return {"error": f"Multiclass prediction failed: {str(e)}"}

            polar = 'NQST'         polar = 'NQST' 

    def predict_comprehensive(self, sequence, binary_threshold=0.5, multiclass_threshold=0.7):

        """Comprehensive prediction using both binary and multiclass models"""        charged = 'KRDEH'        charged = 'KRDEH'

        if not self.validate_sequence(sequence):

            return {"error": "Invalid peptide sequence"}        aromatic = 'FWY'        aromatic = 'FWY'

        

        results = {        tiny = 'ACSV'        tiny = 'ACSV'

            "sequence": sequence,

            "sequence_length": len(sequence),        small = 'ABDHNT'        small = 'ABDHNT'

            "predictions": {}

        }        aliphatic = 'ILV'        aliphatic = 'ILV'

        

        binary_result = self.predict_binary_therapeutic(sequence, binary_threshold)                

        if "error" not in binary_result:

            results["predictions"]["binary_therapeutic"] = binary_result        features.extend([        features.extend([

        

        multiclass_result = self.predict_therapeutic_category(sequence, multiclass_threshold)            sum(seq.count(aa) for aa in hydrophobic) / seq_len,            sum(seq.count(aa) for aa in hydrophobic) / seq_len,

        if "error" not in multiclass_result:

            results["predictions"]["therapeutic_category"] = multiclass_result            sum(seq.count(aa) for aa in polar) / seq_len,            sum(seq.count(aa) for aa in polar) / seq_len,

        

        return results            sum(seq.count(aa) for aa in charged) / seq_len,            sum(seq.count(aa) for aa in charged) / seq_len,

            sum(seq.count(aa) for aa in aromatic) / seq_len,            sum(seq.count(aa) for aa in aromatic) / seq_len,

            sum(seq.count(aa) for aa in tiny) / seq_len,            sum(seq.count(aa) for aa in tiny) / seq_len,

            sum(seq.count(aa) for aa in small) / seq_len,            sum(seq.count(aa) for aa in small) / seq_len,

            sum(seq.count(aa) for aa in aliphatic) / seq_len,            sum(seq.count(aa) for aa in aliphatic) / seq_len,

        ])        ])

                

        # 6. Charge properties        # 6. Charge properties

        positive_charge = seq.count('K') + seq.count('R') + seq.count('H')        positive_charge = seq.count('K') + seq.count('R') + seq.count('H')

        negative_charge = seq.count('D') + seq.count('E')        negative_charge = seq.count('D') + seq.count('E')

        net_charge = positive_charge - negative_charge        net_charge = positive_charge - negative_charge

                

        features.extend([        features.extend([

            positive_charge / seq_len,            positive_charge / seq_len,

            negative_charge / seq_len,            negative_charge / seq_len,

            net_charge / seq_len,            net_charge / seq_len,

            abs(net_charge) / seq_len,            abs(net_charge) / seq_len,

        ])        ])

                

        # 7. Structural features        # 7. Structural features

        proline_content = seq.count('P') / seq_len        proline_content = seq.count('P') / seq_len

        glycine_content = seq.count('G') / seq_len        glycine_content = seq.count('G') / seq_len

        cysteine_content = seq.count('C') / seq_len        cysteine_content = seq.count('C') / seq_len

                

        features.extend([proline_content, glycine_content, cysteine_content])        features.extend([proline_content, glycine_content, cysteine_content])

                

        # 8. Sequence patterns        # 8. Sequence patterns

        max_repeat = 1        max_repeat = 1

        current_repeat = 1        current_repeat = 1

        for i in range(1, seq_len):        for i in range(1, seq_len):

            if seq[i] == seq[i-1]:            if seq[i] == seq[i-1]:

                current_repeat += 1                current_repeat += 1

                max_repeat = max(max_repeat, current_repeat)                max_repeat = max(max_repeat, current_repeat)

            else:            else:

                current_repeat = 1                current_repeat = 1

                

        features.append(max_repeat / seq_len)        features.append(max_repeat / seq_len)

                

        # 9. Hydrophobic moments and amphiphilicity        # 9. Hydrophobic moments and amphiphilicity

        hydrophobic_values = {        hydrophobic_values = {

            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,

            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,

            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,

            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2

        }        }

                

        hydrophobic_moment = 0        hydrophobic_moment = 0

        for i, aa in enumerate(seq):        for i, aa in enumerate(seq):

            if aa in hydrophobic_values:            if aa in hydrophobic_values:

                angle = i * 100 * np.pi / 180  # Assuming alpha-helix (100 degrees per residue)                angle = i * 100 * np.pi / 180  # Assuming alpha-helix (100 degrees per residue)

                hydrophobic_moment += hydrophobic_values[aa] * np.exp(1j * angle)                hydrophobic_moment += hydrophobic_values[aa] * np.exp(1j * angle)

                

        features.append(abs(hydrophobic_moment) / seq_len)        features.append(abs(hydrophobic_moment) / seq_len)

                

        return np.array(features)        return np.array(features)

        

    def validate_sequence(self, sequence):    def validate_sequence(self, sequence):

        """Validate peptide sequence"""        """Validate peptide sequence"""

        if not sequence or not isinstance(sequence, str):        if not sequence or not isinstance(sequence, str):

            return False            return False

                

        # Check for valid amino acids        # Check for valid amino acids

        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')

        sequence_upper = sequence.upper()        sequence_upper = sequence.upper()

                

        if not all(aa in valid_aas for aa in sequence_upper):        if not all(aa in valid_aas for aa in sequence_upper):

            return False            return False

                

        # Check length (reasonable peptide length)        # Check length (reasonable peptide length)

        if len(sequence) < 3 or len(sequence) > 200:        if len(sequence) < 3 or len(sequence) > 200:

            return False            return True

            

        return True    def predict_binary_therapeutic(self, sequence, confidence_threshold=0.5):

            """Predict if a peptide is therapeutic (binary classification)"""

    def predict_binary_therapeutic(self, sequence, confidence_threshold=0.5):        try:

        """Predict if a peptide is therapeutic (binary classification)"""            # Get binary feature model and scaler

        try:            model = self.model_loader.get_model('binary_feature')

            # Get binary feature model and scaler            scaler = self.model_loader.get_scaler('feature_scaler')

            model = self.model_loader.get_model('comprehensive_feature')            

            scaler = self.model_loader.get_scaler('comprehensive_feature_scaler')            if not model or not scaler:

                            return {"error": "Binary classification model not available"}

            if not model or not scaler:            

                return {"error": "Binary classification model not available"}            # Extract features

                        features = self.extract_comprehensive_features(sequence)

            # Extract features            features_scaled = scaler.transform(features.reshape(1, -1))

            features = self.extract_comprehensive_features(sequence)            

            features_scaled = scaler.transform(features.reshape(1, -1))            # Make prediction

                        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]

            # Make prediction            is_therapeutic = prediction_prob > confidence_threshold

            prediction_prob = model.predict(features_scaled, verbose=0)[0][0]            

            is_therapeutic = prediction_prob > confidence_threshold            return {

                            "sequence": sequence,

            return {                "is_therapeutic": bool(is_therapeutic),

                "sequence": sequence,                "confidence": float(prediction_prob),

                "is_therapeutic": bool(is_therapeutic),                "prediction_type": "binary_therapeutic",

                "confidence": float(prediction_prob),                "confidence_threshold": confidence_threshold,

                "prediction_type": "binary_therapeutic",                "model_used": "binary_feature"

                "confidence_threshold": confidence_threshold,            }

                "model_used": "comprehensive_feature"            

            }        except Exception as e:

                        return {"error": f"Binary prediction failed: {str(e)}"}

        except Exception as e:    

            return {"error": f"Binary prediction failed: {str(e)}"}    def predict_therapeutic_category(self, sequence, confidence_threshold=0.7, use_model='multiclass_therapeutic'):

            """Predict therapeutic category (15-class classification)"""

    def predict_therapeutic_category(self, sequence, confidence_threshold=0.7, use_model='multiclass_therapeutic'):        try:

        """Predict therapeutic category (15-class classification)"""            # Get multiclass model and required preprocessors

        try:            model = self.model_loader.get_model(use_model)

            # Get multiclass model and required preprocessors            

            if use_model == 'multiclass_therapeutic':            if not model:

                model = self.model_loader.get_model('therapeutic_peptide_classifier_15class')                return {"error": f"Model '{use_model}' not available"}

                scaler = self.model_loader.get_scaler('therapeutic_scaler_15class')            

                pca = self.model_loader.get_scaler('therapeutic_pca_15class')            # Handle different model types

                            if use_model == 'multiclass_therapeutic':

                if not model or not scaler or not pca:                # Use ProtBERT + PCA pipeline

                    return {"error": "Required components not available for 15-class model"}                scaler = self.model_loader.get_scaler('therapeutic_scaler')

                                pca = self.model_loader.get_scaler('therapeutic_pca')

                # Generate ProtBERT-like features (placeholder - replace with actual ProtBERT features)                

                features = np.random.randn(1, 1024)                if not scaler or not pca:

                features_scaled = scaler.transform(features)                    return {"error": "Required preprocessors not available for therapeutic model"}

                features_pca = pca.transform(features_scaled)                

                features_final = np.expand_dims(features_pca, axis=-1)                # Generate random features as placeholder (replace with actual ProtBERT features)

                                features = np.random.randn(1, 1024)

            elif use_model == 'multiclass_base':                features_scaled = scaler.transform(features)

                model = self.model_loader.get_model('multiclass_therapeutic_peptide')                features_pca = pca.transform(features_scaled)

                scaler = self.model_loader.get_scaler('multiclass_scaler')                features_final = np.expand_dims(features_pca, axis=-1)

                                

                if not model or not scaler:            elif use_model == 'multiclass_base':

                    return {"error": "Required components not available for base multiclass model"}                # Use standard feature extraction

                                scaler = self.model_loader.get_scaler('multiclass_scaler')

                features = self.extract_comprehensive_features(sequence)                

                features_scaled = scaler.transform(features.reshape(1, -1))                if not scaler:

                features_final = features_scaled                    return {"error": "Required scaler not available for base model"}

                                

            else:                features = self.extract_comprehensive_features(sequence)

                return {"error": f"Unknown model type: {use_model}"}                features_scaled = scaler.transform(features.reshape(1, -1))

                            features_final = features_scaled

            # Make prediction                

            prediction_probs = model.predict(features_final, verbose=0)[0]            else:

            predicted_class = np.argmax(prediction_probs)                return {"error": f"Unknown model type: {use_model}"}

            confidence = np.max(prediction_probs)            

                        # Make prediction

            # Get category name            prediction_probs = model.predict(features_final, verbose=0)[0]

            if predicted_class < len(self.category_names):            predicted_class = np.argmax(prediction_probs)

                category = self.category_names[predicted_class]            confidence = np.max(prediction_probs)

            else:            

                category = f"Unknown_Category_{predicted_class}"            # Get category name

                        if predicted_class < len(self.category_names):

            # Determine if prediction meets confidence threshold                category = self.category_names[predicted_class]

            meets_threshold = confidence >= confidence_threshold            else:

                            category = f"Unknown_Category_{predicted_class}"

            # Get top 3 predictions            

            top_indices = np.argsort(prediction_probs)[-3:][::-1]            # Determine if prediction meets confidence threshold

            top_predictions = [            meets_threshold = confidence >= confidence_threshold

                {            

                    "category": self.category_names[idx] if idx < len(self.category_names) else f"Unknown_{idx}",            # Get top 3 predictions

                    "probability": float(prediction_probs[idx])            top_indices = np.argsort(prediction_probs)[-3:][::-1]

                }            top_predictions = [

                for idx in top_indices                {

            ]                    "category": self.category_names[idx] if idx < len(self.category_names) else f"Unknown_{idx}",

                                "probability": float(prediction_probs[idx])

            return {                }

                "sequence": sequence,                for idx in top_indices

                "predicted_category": category if meets_threshold else "Low Confidence - Uncertain",            ]

                "predicted_class": int(predicted_class),            

                "confidence": float(confidence),            return {

                "confidence_threshold": confidence_threshold,                "sequence": sequence,

                "meets_threshold": meets_threshold,                "predicted_category": category if meets_threshold else "Low Confidence - Uncertain",

                "top_predictions": top_predictions,                "predicted_class": int(predicted_class),

                "all_probabilities": {                "confidence": float(confidence),

                    self.category_names[i] if i < len(self.category_names) else f"Unknown_{i}": float(prob)                "confidence_threshold": confidence_threshold,

                    for i, prob in enumerate(prediction_probs)                "meets_threshold": meets_threshold,

                },                "top_predictions": top_predictions,

                "prediction_type": "multiclass_therapeutic",                "all_probabilities": {

                "model_used": use_model                    self.category_names[i] if i < len(self.category_names) else f"Unknown_{i}": float(prob)

            }                    for i, prob in enumerate(prediction_probs)

                            },

        except Exception as e:                "prediction_type": "multiclass_therapeutic",

            return {"error": f"Multiclass prediction failed: {str(e)}"}                "model_used": use_model

                }

    def predict_comprehensive(self, sequence, binary_threshold=0.5, multiclass_threshold=0.7):            

        """Comprehensive prediction using both binary and multiclass models"""        except Exception as e:

        try:            return {"error": f"Multiclass prediction failed: {str(e)}"}

            # Validate sequence    

            if not self.validate_sequence(sequence):    def predict_comprehensive(self, sequence, binary_threshold=0.5, multiclass_threshold=0.7):

                return {"error": "Invalid peptide sequence"}        """Comprehensive prediction using both binary and multiclass models"""

                    try:

            results = {            # Validate sequence

                "sequence": sequence,            if not self.validate_sequence(sequence):

                "sequence_length": len(sequence),                return {"error": "Invalid peptide sequence"}

                "timestamp": str(np.datetime64('now')),            

                "predictions": {}            results = {

            }                "sequence": sequence,

                            "sequence_length": len(sequence),

            # Binary therapeutic prediction                "timestamp": str(np.datetime64('now')),

            binary_result = self.predict_binary_therapeutic(sequence, binary_threshold)                "predictions": {}

            if "error" not in binary_result:            }

                results["predictions"]["binary_therapeutic"] = binary_result            

                        # Binary therapeutic prediction

            # Multiclass therapeutic category prediction            binary_result = self.predict_binary_therapeutic(sequence, binary_threshold)

            multiclass_result = self.predict_therapeutic_category(sequence, multiclass_threshold)            if "error" not in binary_result:

            if "error" not in multiclass_result:                results["predictions"]["binary_therapeutic"] = binary_result

                results["predictions"]["therapeutic_category"] = multiclass_result            

                        # Multiclass therapeutic category prediction

            # Summary recommendation            multiclass_result = self.predict_therapeutic_category(sequence, multiclass_threshold)

            if ("binary_therapeutic" in results["predictions"] and             if "error" not in multiclass_result:

                "therapeutic_category" in results["predictions"]):                results["predictions"]["therapeutic_category"] = multiclass_result

                            

                binary_therapeutic = results["predictions"]["binary_therapeutic"]["is_therapeutic"]            # Summary recommendation

                category_confident = results["predictions"]["therapeutic_category"]["meets_threshold"]            if ("binary_therapeutic" in results["predictions"] and 

                                "therapeutic_category" in results["predictions"]):

                if binary_therapeutic and category_confident:                

                    recommendation = "Strong therapeutic potential with specific category identified"                binary_therapeutic = results["predictions"]["binary_therapeutic"]["is_therapeutic"]

                elif binary_therapeutic:                category_confident = results["predictions"]["therapeutic_category"]["meets_threshold"]

                    recommendation = "Likely therapeutic but category uncertain"                

                elif category_confident:                if binary_therapeutic and category_confident:

                    recommendation = "Category identified but therapeutic potential uncertain"                    recommendation = "Strong therapeutic potential with specific category identified"

                else:                elif binary_therapeutic:

                    recommendation = "Low confidence in therapeutic potential"                    recommendation = "Likely therapeutic but category uncertain"

                                elif category_confident:

                results["recommendation"] = recommendation                    recommendation = "Category identified but therapeutic potential uncertain"

                            else:

            return results                    recommendation = "Low confidence in therapeutic potential"

                            

        except Exception as e:                results["recommendation"] = recommendation

            return {"error": f"Comprehensive prediction failed: {str(e)}"}            

                return results

    def get_model_info(self):            

        """Get information about loaded models"""        except Exception as e:

        return {            return {"error": f"Comprehensive prediction failed: {str(e)}"}

            "available_models": self.model_loader.list_available_models(),    

            "available_scalers": self.model_loader.list_available_scalers(),    def get_model_info(self):

            "category_names": self.category_names,        """Get information about loaded models"""

            "model_details": self.model_loader.get_model_info()        return {

        }            "available_models": self.model_loader.list_available_models(),
            "available_scalers": self.model_loader.list_available_scalers(),
            "category_names": self.category_names,
            "model_details": self.model_loader.get_model_info()
        }

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