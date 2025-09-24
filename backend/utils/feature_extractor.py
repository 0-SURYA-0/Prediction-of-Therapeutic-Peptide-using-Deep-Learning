"""
Feature Extraction Module for Therapeutic Peptide Prediction
Implements comprehensive biochemical feature extraction as used in research notebooks
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    logging.warning("BioPython not available. Some features will be disabled.")

logger = logging.getLogger(__name__)


class PeptideFeatureExtractor:
    """
    Comprehensive peptide feature extraction based on biochemical properties.
    Implements the 65+ feature extraction pipeline from the research notebooks.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.hydrophobic_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        # Amino acid groups
        self.aa_groups = {
            'hydrophobic': 'AILMFWV',
            'polar': 'NQST',
            'charged': 'KRDEH',
            'aromatic': 'FWY',
            'tiny': 'ACSV',
            'small': 'ABDHNT',
            'aliphatic': 'ILV'
        }
    
    def extract_comprehensive_features(self, sequence: str) -> np.ndarray:
        """
        Extract comprehensive biochemical features from peptide sequence.
        This implements the 65+ feature extraction pipeline.
        
        Args:
            sequence (str): Peptide sequence using single-letter amino acid codes
            
        Returns:
            np.ndarray: Array of extracted features
        """
        sequence = sequence.upper().strip()
        if not self.validate_sequence(sequence):
            raise ValueError(f"Invalid peptide sequence: {sequence}")
        
        features = []
        seq_length = len(sequence)
        
        # 1. Sequence length
        features.append(seq_length)
        
        # 2. Amino acid composition (20 features)
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        for aa in self.amino_acids:
            features.append(aa_counts[aa] / seq_length)
        
        # 3. Dipeptide composition (selected important dipeptides)
        dipeptides = [
            'AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL',
            'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL'
        ]
        dipeptide_counts = {}
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            dipeptide_counts[dipeptide] = dipeptide_counts.get(dipeptide, 0) + 1
        
        for dipeptide in dipeptides:
            count = dipeptide_counts.get(dipeptide, 0)
            features.append(count / (seq_length - 1) if seq_length > 1 else 0)
        
        # 4. Amino acid group compositions
        for group_name, group_aas in self.aa_groups.items():
            group_count = sum(aa_counts.get(aa, 0) for aa in group_aas)
            features.append(group_count / seq_length)
        
        # 5. Charge properties
        positive_charge = aa_counts.get('K', 0) + aa_counts.get('R', 0) + aa_counts.get('H', 0)
        negative_charge = aa_counts.get('D', 0) + aa_counts.get('E', 0)
        net_charge = positive_charge - negative_charge
        
        features.extend([
            positive_charge / seq_length,
            negative_charge / seq_length,
            net_charge / seq_length,
            abs(net_charge) / seq_length
        ])
        
        # 6. Structural properties
        features.extend([
            aa_counts.get('P', 0) / seq_length,  # Proline content
            aa_counts.get('G', 0) / seq_length,  # Glycine content
            aa_counts.get('C', 0) / seq_length   # Cysteine content
        ])
        
        # 7. Hydrophobic moment
        hydrophobic_moment = 0
        for i, aa in enumerate(sequence):
            if aa in self.hydrophobic_scale:
                angle = i * 100 * np.pi / 180  # Alpha-helix angle
                hydrophobic_moment += self.hydrophobic_scale[aa] * np.exp(1j * angle)
        features.append(abs(hydrophobic_moment) / seq_length)
        
        # 8. BioPython features (if available)
        if HAS_BIOPYTHON:
            try:
                analyzer = ProteinAnalysis(sequence)
                features.extend([
                    analyzer.molecular_weight(),
                    analyzer.aromaticity(),
                    analyzer.instability_index(),
                    analyzer.isoelectric_point(),
                    analyzer.gravy()  # Grand Average of Hydropathy
                ])
                
                # Secondary structure fractions
                sec_struct = analyzer.secondary_structure_fraction()
                features.extend([sec_struct[0], sec_struct[1], sec_struct[2]])
                
            except Exception as e:
                logger.warning(f"BioPython analysis failed for sequence {sequence}: {str(e)}")
                # Add zeros for missing features
                features.extend([0] * 8)
        else:
            # Add zeros for missing BioPython features
            features.extend([0] * 8)
        
        return np.array(features, dtype=np.float32)
    
    def extract_basic_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract basic biochemical features for Model 3 (biological properties).
        
        Args:
            sequence (str): Peptide sequence
            
        Returns:
            dict: Dictionary of feature names and values
        """
        if not HAS_BIOPYTHON:
            logger.warning("BioPython not available. Using placeholder values.")
            return {
                'Molecular Weight': len(sequence) * 110,  # Rough estimate
                'Aromaticity': 0.0,
                'Instability Index': 40.0,
                'Isoelectric Point': 7.0,
                'Hydrophobicity (GRAVY)': 0.0
            }
        
        try:
            analyzer = ProteinAnalysis(sequence)
            return {
                'Molecular Weight': analyzer.molecular_weight(),
                'Aromaticity': analyzer.aromaticity(),
                'Instability Index': analyzer.instability_index(),
                'Isoelectric Point': analyzer.isoelectric_point(),
                'Hydrophobicity (GRAVY)': analyzer.gravy()
            }
        except Exception as e:
            logger.error(f"Error extracting basic features: {str(e)}")
            return {
                'Molecular Weight': 0.0,
                'Aromaticity': 0.0,
                'Instability Index': 0.0,
                'Isoelectric Point': 0.0,
                'Hydrophobicity (GRAVY)': 0.0
            }
    
    def extract_statistical_features(self, sequence: str) -> List[float]:
        """
        Extract statistical features as used in CNN-LSTM model.
        
        Args:
            sequence (str): Peptide sequence
            
        Returns:
            list: List of statistical features
        """
        sequence = sequence.upper().strip()
        seq_length = len(sequence)
        
        if seq_length == 0:
            return [0] * 15
        
        # Amino acid counts
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        
        # Group compositions
        features = []
        for group_name, group_aas in self.aa_groups.items():
            group_fraction = sum(aa_counts.get(aa, 0) for aa in group_aas) / seq_length
            features.append(group_fraction)
        
        # Charge properties
        positive_charge = sum(aa_counts.get(aa, 0) for aa in 'KRH')
        negative_charge = sum(aa_counts.get(aa, 0) for aa in 'DE')
        net_charge = positive_charge - negative_charge
        
        features.extend([
            positive_charge / seq_length,
            negative_charge / seq_length,
            net_charge / seq_length,
            abs(net_charge) / seq_length
        ])
        
        # Structural features
        features.extend([
            aa_counts.get('P', 0) / seq_length,  # Proline
            aa_counts.get('G', 0) / seq_length,  # Glycine
            aa_counts.get('C', 0) / seq_length   # Cysteine
        ])
        
        # Hydrophobic moment
        hydrophobic_moment = 0
        for i, aa in enumerate(sequence):
            if aa in self.hydrophobic_scale:
                angle = i * 100 * np.pi / 180
                hydrophobic_moment += self.hydrophobic_scale[aa] * np.exp(1j * angle)
        features.append(abs(hydrophobic_moment) / seq_length)
        
        return features
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate if sequence contains only valid amino acids.
        
        Args:
            sequence (str): Peptide sequence to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not sequence or not isinstance(sequence, str):
            return False
        
        sequence = sequence.upper().strip()
        
        # Check if all characters are valid amino acids
        if not all(aa in self.amino_acids for aa in sequence):
            return False
        
        # Check reasonable length constraints
        if len(sequence) < 3 or len(sequence) > 200:
            return False
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features extracted by extract_comprehensive_features.
        
        Returns:
            list: List of feature names
        """
        names = ['Length']
        
        # Amino acid composition
        names.extend([f'{aa}_composition' for aa in self.amino_acids])
        
        # Dipeptide composition
        dipeptides = [
            'AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL',
            'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL'
        ]
        names.extend([f'{dp}_dipeptide' for dp in dipeptides])
        
        # Group compositions
        names.extend([f'{group}_fraction' for group in self.aa_groups.keys()])
        
        # Charge properties
        names.extend([
            'positive_charge_fraction',
            'negative_charge_fraction', 
            'net_charge_fraction',
            'absolute_charge_fraction'
        ])
        
        # Structural properties
        names.extend(['proline_content', 'glycine_content', 'cysteine_content'])
        
        # Hydrophobic moment
        names.append('hydrophobic_moment')
        
        # BioPython features
        names.extend([
            'molecular_weight',
            'aromaticity', 
            'instability_index',
            'isoelectric_point',
            'gravy',
            'helix_fraction',
            'turn_fraction',
            'sheet_fraction'
        ])
        
        return names