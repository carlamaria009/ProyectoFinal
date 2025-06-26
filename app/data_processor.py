import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import json

class DataProcessor:
    """Class to handle data processing for ML model input and output"""
    
    @staticmethod
    def parse_input_data(input_text: str, expected_features: Optional[List[str]] = None) -> Tuple[bool, Any, str]:
        """
        Parse input text into data suitable for model prediction
        
        Tries multiple formats: JSON, CSV, space-separated
        """
        try:
            input_text = input_text.strip()
            
            if not input_text:
                return False, None, "Input data is empty"
            
            # Try to parse as JSON first
            try:
                data = json.loads(input_text)
                return DataProcessor._process_json_data(data, expected_features)
            except json.JSONDecodeError:
                pass
            
            # Try to parse as comma-separated values
            try:
                return DataProcessor._process_csv_data(input_text, expected_features)
            except Exception:
                pass
            
            # Try to parse as space-separated values
            try:
                return DataProcessor._process_space_separated_data(input_text, expected_features)
            except Exception:
                pass
            
            return False, None, "Unable to parse input data. Please use JSON format, comma-separated values, or space-separated values."
            
        except Exception as e:
            return False, None, f"Error parsing input data: {str(e)}"
    
    @staticmethod
    def _process_json_data(data: Any, expected_features: Optional[List[str]] = None) -> Tuple[bool, Any, str]:
        """Process JSON data - handles single samples and multiple samples"""
        try:
            if isinstance(data, dict):
                # Single sample as dictionary
                if expected_features:
                    missing_features = set(expected_features) - set(data.keys())
                    if missing_features:
                        return False, None, f"Missing features: {list(missing_features)}"
                    
                    ordered_data = [data[feature] 
                                    for feature in expected_features]
                    return True, np.array([ordered_data]), "JSON data processed successfully"
                else:
                    return True, np.array([list(data.values())]), "JSON data processed successfully"
            
            elif isinstance(data, list):
                if len(data) == 0:
                    return False, None, "Empty data list"
                
                # Check if it's a list of dictionaries (multiple samples)
                if isinstance(data[0], dict):
                    if expected_features:
                        processed_data = []
                        for sample in data:
                            missing_features = set(expected_features) - set(sample.keys())
                            if missing_features:
                                return False, None, f"Missing features in sample: {list(missing_features)}"
                            ordered_sample = [sample[feature] for feature in expected_features]
                            processed_data.append(ordered_sample)
                        return True, np.array(processed_data), "JSON data processed successfully"
                    else:
                        processed_data = [list(sample.values()) for sample in data]
                        return True, np.array(processed_data), "JSON data processed successfully"
                
                # List of values (single sample)
                else:
                    return True, np.array([data]), "JSON data processed successfully"
            
        except Exception as e:
            return False, None, f"Error processing JSON data: {str(e)}"
    
    @staticmethod
    def format_predictions_for_display(predictions: Any, probabilities: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Format predictions for display in table
        
        Creates a structured format with sample numbers, predictions, and probabilities
        """
        try:
            formatted_data = []
            
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            if probabilities is not None and not isinstance(probabilities, np.ndarray):
                probabilities = np.array(probabilities)
            
            # Process each prediction
            for i, pred in enumerate(predictions):
                row = {
                    'Sample': i + 1,
                    'Prediction': str(pred)
                }
                
                # Add probabilities if available
                if probabilities is not None:
                    if probabilities.ndim == 2:  # Multiple classes
                        for j, prob in enumerate(probabilities[i]):
                            row[f'Probability_Class_{j}'] = f"{prob:.4f}"
                    else:  # Single probability
                        row['Probability'] = f"{probabilities[i]:.4f}"
                
                formatted_data.append(row)
            
            return formatted_data
            
        except Exception as e:
            return [{'Error': f"Failed to format predictions: {str(e)}"}]