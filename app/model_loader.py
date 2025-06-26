import pickle
import joblib
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class ModelLoader:
    """Class to handle loading and executing PKL machine learning models"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_info = {}

    def load_model(self, file_path: str) -> Tuple[bool, str]:
        """
        Load a PKL model from file

        Args:
            file_path (str): Path to the PKL file

        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"

            if not file_path.lower().endswith(('.pkl', '.pickle', '.joblib')):
                return False, "Invalid file format. Please select a .pkl, .pickle, or .joblib file"

            # Try loading with pickle first
            try:
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
                loader_used = "pickle"
            except Exception:
                # If pickle fails, try joblib
                try:
                    self.model = joblib.load(file_path)
                    loader_used = "joblib"
                except Exception as e:
                    return False, f"Failed to load model with both pickle and joblib: {str(e)}"

            self.model_path = file_path

            # Extract model information
            self._extract_model_info(loader_used)

            return True, f"Model loaded successfully using {loader_used}"

        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def _extract_model_info(self, loader_used: str):
        """Extract information about the loaded model"""
        self.model_info = {
            'file_path': self.model_path,
            'loader_used': loader_used,
            'model_type': type(self.model).__name__,
            'model_module': type(self.model).__module__,
        }

        # Try to extract additional information based on model type
        try:
            if hasattr(self.model, 'feature_names_in_'):
                self.model_info['feature_names'] = list(
                    self.model.feature_names_in_
                )
            elif hasattr(self.model, 'feature_importances_'):
                self.model_info['n_features'] = len(
                    self.model.feature_importances_
                )
            elif hasattr(self.model, 'coef_'):
                if self.model.coef_.ndim == 1:
                    self.model_info['n_features'] = len(self.model.coef_)
                else:
                    self.model_info['n_features'] = self.model.coef_.shape[1]
        except Exception:
            pass

    def predict(self, input_data: Any) -> Tuple[bool, Any, str]:
        """Make predictions using the loaded model"""
        try:
            if self.model is None:
                return False, None, "No model loaded"

            predictions = self.model.predict(input_data)
            return True, predictions, "Prediction successful"

        except Exception as e:
            return False, None, f"Prediction error: {str(e)}"

    def predict_proba(self, input_data: Any) -> Tuple[bool, Any, str]:
        """Get prediction probabilities if available"""
        try:
            if self.model is None:
                return False, None, "No model loaded"

            if not hasattr(self.model, 'predict_proba'):
                return False, None, "Model does not support probability predictions"

            probabilities = self.model.predict_proba(input_data)
            return True, probabilities, "Probability prediction successful"

        except Exception as e:
            return False, None, f"Probability prediction error: {str(e)}"
