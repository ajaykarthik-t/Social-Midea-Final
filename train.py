#!/usr/bin/env python3
"""
train.py - Simple model training for fake social media account detection.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


class SimpleModelTrainer:
    """Class for training models for fake account detection."""
    
    def __init__(self, data_dir='data', output_dir='models'):
        """Initialize the model trainer."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    def load_data(self):
        """
        Load the preprocessed data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Loading data...")
        
        # Define file paths
        train_features_path = os.path.join(self.data_dir, "train_features.npy")
        test_features_path = os.path.join(self.data_dir, "test_features.npy")
        train_data_path = os.path.join(self.data_dir, "train_data.csv")
        test_data_path = os.path.join(self.data_dir, "test_data.csv")
        
        # Check if all files exist
        for path in [train_features_path, test_features_path, train_data_path, test_data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load data
        X_train = np.load(train_features_path)
        X_test = np.load(test_features_path)
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        # Extract labels
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Loaded testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return X_train, X_test, y_train, y_test, train_df, test_df
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train a logistic regression model.
        
        Returns:
            Trained model
        """
        print("Training logistic regression model...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train a random forest model.
        
        Returns:
            Trained model
        """
        print("Training random forest model...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a trained model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {model_name} model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print metrics
        print(f"Model: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"Confusion matrix:\n{cm}")
        
        return metrics
    
    def save_model(self, model, model_name):
        """
        Save a trained model to disk.
        
        Returns:
            Path to the saved model
        """
        print(f"Saving {model_name} model...")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Save model info
        info_path = os.path.join(self.output_dir, f"{model_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'model_type': model.__class__.__name__,
                'features': model.n_features_in_,
                'classes': model.classes_.tolist()
            }, f)
        
        return model_path


def main():
    """Main function to train and evaluate models."""
    # Create a model trainer
    trainer = SimpleModelTrainer(data_dir='data', output_dir='models')
    
    # Load data
    X_train, X_test, y_train, y_test, _, _ = trainer.load_data()
    
    # Train logistic regression model
    logistic_model = trainer.train_logistic_regression(X_train, y_train)
    
    # Evaluate logistic regression model
    trainer.evaluate_model(logistic_model, X_test, y_test, 'logistic')
    
    # Save logistic regression model
    trainer.save_model(logistic_model, 'logistic')
    
    # Train random forest model
    rf_model = trainer.train_random_forest(X_train, y_train)
    
    # Evaluate random forest model
    trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    
    # Save random forest model
    trainer.save_model(rf_model, 'random_forest')
    
    print("Model training and evaluation completed!")


if __name__ == "__main__":
    main()