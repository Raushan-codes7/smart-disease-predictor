"""
Data preprocessing module for disease prediction system.
Handles data loading, cleaning, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


def create_synthetic_dataset():
    """
    Create a synthetic disease-symptom dataset for demonstration.
    Returns a DataFrame with diseases and their corresponding symptoms.
    """
    # Define diseases and their typical symptoms
    diseases_data = {
        'Common Cold': ['runny_nose', 'sneezing', 'cough', 'sore_throat', 'fatigue', 'mild_fever'],
        'Flu': ['high_fever', 'body_aches', 'fatigue', 'cough', 'headache', 'chills', 'sore_throat'],
        'Migraine': ['severe_headache', 'nausea', 'light_sensitivity', 'sound_sensitivity', 'dizziness'],
        'Diabetes': ['frequent_urination', 'excessive_thirst', 'weight_loss', 'fatigue', 'blurred_vision'],
        'Hypertension': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath', 'fatigue'],
        'Asthma': ['wheezing', 'shortness_of_breath', 'chest_tightness', 'cough', 'fatigue'],
        'Pneumonia': ['cough', 'fever', 'shortness_of_breath', 'chest_pain', 'fatigue', 'sweating'],
        'Bronchitis': ['cough', 'mucus_production', 'chest_discomfort', 'fatigue', 'slight_fever'],
        'Gastritis': ['stomach_pain', 'nausea', 'vomiting', 'bloating', 'loss_of_appetite'],
        'Arthritis': ['joint_pain', 'joint_stiffness', 'swelling', 'reduced_mobility', 'fatigue'],
        'Depression': ['sadness', 'fatigue', 'sleep_problems', 'loss_of_interest', 'appetite_changes'],
        'Anxiety': ['excessive_worry', 'restlessness', 'fatigue', 'sleep_problems', 'muscle_tension'],
        'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_intolerance', 'depression', 'dry_skin'],
        'Hyperthyroidism': ['weight_loss', 'rapid_heartbeat', 'anxiety', 'heat_intolerance', 'fatigue'],
        'COPD': ['shortness_of_breath', 'chronic_cough', 'wheezing', 'chest_tightness', 'fatigue']
    }
    
    # Get all unique symptoms
    all_symptoms = set()
    for symptoms in diseases_data.values():
        all_symptoms.update(symptoms)
    all_symptoms = sorted(list(all_symptoms))
    
    # Create DataFrame
    data = []
    for disease, symptoms in diseases_data.items():
        # Create multiple samples per disease with slight variations
        for _ in range(50):  # 50 samples per disease
            row = {'disease': disease}
            # Add primary symptoms (always present)
            for symptom in symptoms:
                row[symptom] = 1
            
            # Add some secondary symptoms randomly (simulate real-world variability)
            for symptom in all_symptoms:
                if symptom not in row:
                    # 10% chance of having a secondary symptom
                    row[symptom] = 1 if np.random.random() < 0.1 else 0
            
            data.append(row)
    
    # Create DataFrame and fill missing values
    df = pd.DataFrame(data)
    df = df.fillna(0)
    
    return df


def load_data(data_path=None):
    """
    Load disease-symptom dataset from CSV file or create synthetic data.
    
    Args:
        data_path (str): Path to CSV file. If None, creates synthetic data.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Creating synthetic dataset...")
        df = create_synthetic_dataset()
        
        # Save synthetic dataset for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/synthetic_disease_data.csv', index=False)
        print("Synthetic dataset saved to data/synthetic_disease_data.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Diseases: {df['disease'].nunique()}")
    print(f"Symptoms: {df.shape[1] - 1}")
    
    return df


def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and encoding.
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        tuple: (X, y, label_encoder, feature_names)
    """
    print("Preprocessing data...")
    
    # Separate features (symptoms) and target (disease)
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    # Handle missing values - fill with 0 (symptom not present)
    X = X.fillna(0)
    
    # Convert all columns to binary (0 or 1)
    X = (X > 0).astype(int)
    
    # Encode disease labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    feature_names = X.columns.tolist()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_encoded.shape}")
    print(f"Classes: {len(label_encoder.classes_)}")
    
    return X, y_encoded, label_encoder, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, label_encoder, feature_names, output_dir='models'):
    """
    Save preprocessed data and encoders for later use.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        label_encoder: Fitted label encoder
        feature_names: List of feature names
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    # Save encoders and feature names
    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(f'{output_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"Preprocessed data saved to {output_dir}/")


def load_preprocessed_data(data_dir='models'):
    """
    Load preprocessed data and encoders.
    
    Args:
        data_dir: Directory containing saved files
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder, feature_names)
    """
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    with open(f'{data_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Preprocessed data loaded from {data_dir}/")
    return X_train, X_test, y_train, y_test, label_encoder, feature_names


if __name__ == "__main__":
    # Test the preprocessing pipeline
    df = load_data()
    X, y, label_encoder, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_preprocessed_data(X_train, X_test, y_train, y_test, label_encoder, feature_names)
