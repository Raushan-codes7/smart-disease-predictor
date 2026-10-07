"""
Model training module for disease prediction system.
Handles Random Forest training with hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import time


def train_model(X_train, y_train, model_type='random_forest', optimize_hyperparams=True):
    """
    Train a machine learning model for disease prediction.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_type (str): Type of model to train ('random_forest')
        optimize_hyperparams (bool): Whether to optimize hyperparameters
    
    Returns:
        tuple: (trained_model, best_params, cv_scores)
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'random_forest':
        # Base Random Forest model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        if optimize_hyperparams:
            # Hyperparameter grid for optimization
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            print("Optimizing hyperparameters...")
            start_time = time.time()
            
            # Use RandomizedSearchCV for faster optimization
            grid_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,  # Number of parameter settings sampled
                cv=5,  # 5-fold cross-validation
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            end_time = time.time()
            print(f"Hyperparameter optimization completed in {end_time - start_time:.2f} seconds")
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_scores = grid_search.cv_results_
            
            print(f"Best parameters: {best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            best_model = base_model.fit(X_train, y_train)
            best_params = {}
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return best_model, best_params, cv_scores


def evaluate_model_performance(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        label_encoder: Label encoder for disease names
    
    Returns:
        dict: Performance metrics
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Store performance metrics
    performance = {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    return performance


def save_model(model, label_encoder, feature_names, performance, model_dir='models'):
    """
    Save trained model and related information.
    
    Args:
        model: Trained model
        label_encoder: Label encoder
        feature_names: List of feature names
        performance: Model performance metrics
        model_dir: Directory to save files
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = f'{model_dir}/disease_predictor.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    encoder_path = f'{model_dir}/label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature names
    features_path = f'{model_dir}/feature_names.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save performance metrics
    performance_path = f'{model_dir}/performance_metrics.pkl'
    with open(performance_path, 'wb') as f:
        pickle.dump(performance, f)
    
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    print(f"Feature names saved to {features_path}")
    print(f"Performance metrics saved to {performance_path}")


def load_model(model_dir='models'):
    """
    Load trained model and related information.
    
    Args:
        model_dir: Directory containing saved files
    
    Returns:
        tuple: (model, label_encoder, feature_names, performance)
    """
    # Load model
    model_path = f'{model_dir}/disease_predictor.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoder
    encoder_path = f'{model_dir}/label_encoder.pkl'
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load feature names
    features_path = f'{model_dir}/feature_names.pkl'
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load performance metrics
    performance_path = f'{model_dir}/performance_metrics.pkl'
    with open(performance_path, 'rb') as f:
        performance = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model, label_encoder, feature_names, performance


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get and display feature importance from trained model.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to display
    
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} most important symptoms:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def predict_disease(model, label_encoder, feature_names, symptoms_list):
    """
    Predict disease from list of symptoms.
    
    Args:
        model: Trained model
        label_encoder: Label encoder
        feature_names: List of feature names
        symptoms_list: List of symptom names
    
    Returns:
        tuple: (predicted_disease, probabilities_dict)
    """
    # Create feature vector
    feature_vector = np.zeros(len(feature_names))
    
    for symptom in symptoms_list:
        if symptom in feature_names:
            idx = feature_names.index(symptom)
            feature_vector[idx] = 1
    
    # Reshape for prediction
    feature_vector = feature_vector.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    
    # Get disease name
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    
    # Create probabilities dictionary
    disease_names = label_encoder.classes_
    probabilities_dict = {
        disease: prob for disease, prob in zip(disease_names, probabilities)
    }
    
    # Sort by probability
    probabilities_dict = dict(sorted(probabilities_dict.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    return predicted_disease, probabilities_dict


if __name__ == "__main__":
    # Test the training pipeline
    from data_preprocessing import load_preprocessed_data
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_preprocessed_data()
    
    # Train model
    model, best_params, cv_scores = train_model(X_train, y_train, optimize_hyperparams=True)
    
    # Evaluate model
    performance = evaluate_model_performance(model, X_test, y_test, label_encoder)
    
    # Save model
    save_model(model, label_encoder, feature_names, performance)
    
    # Show feature importance
    get_feature_importance(model, feature_names)
    
    # Test prediction
    test_symptoms = ['fever', 'cough', 'fatigue']
    predicted_disease, probabilities = predict_disease(model, label_encoder, feature_names, test_symptoms)
    print(f"\nTest prediction for symptoms {test_symptoms}:")
    print(f"Predicted disease: {predicted_disease}")
    print("Top 3 probabilities:")
    for i, (disease, prob) in enumerate(list(probabilities.items())[:3]):
        print(f"{i+1}. {disease}: {prob:.4f}")
