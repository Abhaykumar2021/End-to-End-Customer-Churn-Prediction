import os
import sys
import yaml
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException


def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file and return its content as dictionary
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        dict: Content of YAML file
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def write_yaml_file(file_path: str, content: dict) -> None:
    """
    Write content to YAML file
    
    Args:
        file_path: Path to the YAML file
        content: Dictionary content to write
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save object to file using dill
    
    Args:
        file_path: Path to save the object
        obj: Object to save
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load object from file using dill
    
    Args:
        file_path: Path to the object file
    
    Returns:
        object: Loaded object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array to file
    
    Args:
        file_path: Path to save the array
        array: Numpy array to save
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy array saved to {file_path}")
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array from file
    
    Args:
        file_path: Path to the array file
    
    Returns:
        np.ndarray: Loaded numpy array
    """
    try:
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)
        logger.info(f"Numpy array loaded from {file_path}")
        return array
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Evaluate multiple models with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        models: Dictionary of models
        param: Dictionary of parameters for grid search
    
    Returns:
        dict: Dictionary of model scores
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}")
            
            # Get parameters for this model
            model_params = param.get(model_name, {})
            
            if model_params:
                # Perform grid search
                gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                logger.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                # Train with default parameters
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted')
            test_recall = recall_score(y_test, y_pred, average='weighted')
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            report[model_name] = {
                'model': model,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            }
            
            logger.info(f"{model_name} - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        return report
    
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e


def get_best_model(model_report: dict, threshold: float = 0.6):
    """
    Get the best model from the model report
    
    Args:
        model_report: Dictionary containing model evaluation results
        threshold: Minimum accuracy threshold
    
    Returns:
        tuple: (best_model_name, best_model_object, best_model_score)
    """
    try:
        # Sort models by accuracy
        sorted_models = sorted(
            model_report.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        best_model_name, best_model_info = sorted_models[0]
        best_model_score = best_model_info['accuracy']
        
        if best_model_score < threshold:
            raise Exception(f"No model found with accuracy > {threshold}")
        
        best_model = best_model_info['model']
        
        logger.info(f"Best model: {best_model_name} with accuracy: {best_model_score:.4f}")
        
        return best_model_name, best_model, best_model_score
    
    except Exception as e:
        raise ChurnPredictionException(e, sys) from e
