"""
Main pipeline orchestrator for the disease prediction system.
Executes the complete ML pipeline: data preprocessing → training → evaluation.
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, preprocess_data, split_data, save_preprocessed_data
from model_training import train_model, evaluate_model_performance, save_model, get_feature_importance
from model_evaluation import generate_evaluation_report


def print_header():
    """Print a formatted header for the pipeline."""
    print("=" * 60)
    print("🏥 SMART DISEASE PREDICTOR - ML PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_step(step_num, title, description=""):
    """Print a formatted step header."""
    print(f"\n{'='*20} STEP {step_num}: {title} {'='*20}")
    if description:
        print(f"Description: {description}")
    print()


def main():
    """Execute the complete machine learning pipeline."""
    start_time = time.time()
    
    # Print header
    print_header()
    
    try:
        # Step 1: Data Loading and Preprocessing
        print_step(1, "DATA PREPROCESSING", "Loading dataset and preparing features")
        
        # Load data
        df = load_data()
        print(f"✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Preprocess data
        X, y, label_encoder, feature_names = preprocess_data(df)
        print(f"✓ Data preprocessed: {X.shape[1]} symptoms, {len(label_encoder.classes_)} diseases")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("✓ Data split into training and test sets")
        
        # Save preprocessed data
        save_preprocessed_data(X_train, X_test, y_train, y_test, label_encoder, feature_names)
        print("✓ Preprocessed data saved")
        
        # Step 2: Model Training
        print_step(2, "MODEL TRAINING", "Training Random Forest with hyperparameter optimization")
        
        # Train model with hyperparameter optimization
        model, best_params, cv_scores = train_model(
            X_train, y_train, 
            model_type='random_forest', 
            optimize_hyperparams=True
        )
        print("✓ Model trained successfully")
        
        # Step 3: Model Evaluation
        print_step(3, "MODEL EVALUATION", "Evaluating model performance on test set")
        
        # Evaluate model performance
        performance = evaluate_model_performance(model, X_test, y_test, label_encoder)
        print(f"✓ Model evaluation completed - Accuracy: {performance['accuracy']:.4f}")
        
        # Step 4: Save Model and Results
        print_step(4, "SAVING MODEL", "Saving trained model and performance metrics")
        
        # Save model and related components
        save_model(model, label_encoder, feature_names, performance)
        print("✓ Model and components saved")
        
        # Step 5: Generate Comprehensive Evaluation Report
        print_step(5, "GENERATING REPORTS", "Creating detailed evaluation report with visualizations")
        
        # Generate predictions for evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Generate comprehensive evaluation report
        metrics = generate_evaluation_report(
            y_test, y_pred, y_pred_proba, label_encoder,
            model_name="Random Forest Disease Predictor"
        )
        print("✓ Evaluation report generated with visualizations")
        
        # Step 6: Display Feature Importance
        print_step(6, "FEATURE ANALYSIS", "Analyzing most important symptoms")
        
        # Display feature importance
        importance_df = get_feature_importance(model, feature_names, top_n=20)
        if importance_df is not None:
            print("✓ Feature importance analysis completed")
        
        # Step 7: Pipeline Summary
        print_step(7, "PIPELINE SUMMARY", "Final results and next steps")
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"🎯 Final model accuracy: {performance['accuracy']:.4f} ({performance['accuracy']:.1%})")
        print(f"📊 Number of diseases: {len(label_encoder.classes_)}")
        print(f"🔬 Number of symptoms: {len(feature_names)}")
        print(f"📈 Training samples: {X_train.shape[0]}")
        print(f"🧪 Test samples: {X_test.shape[0]}")
        
        # Print next steps
        print("\n📋 NEXT STEPS:")
        print("1. Run the Streamlit app: streamlit run src/app.py")
        print("2. Check the 'results/' directory for evaluation visualizations")
        print("3. Review 'models/' directory for saved model components")
        print("4. Examine 'data/' directory for the processed dataset")
        
        # Check if accuracy target is met
        if performance['accuracy'] >= 0.90:
            print(f"\n✅ SUCCESS: Model achieved target accuracy of ≥90% ({performance['accuracy']:.1%})")
        else:
            print(f"\n⚠️  WARNING: Model accuracy ({performance['accuracy']:.1%}) is below 90% target")
            print("   Consider: more data, feature engineering, or different algorithms")
        
        print("\n" + "=" * 60)
        print("🏥 SMART DISEASE PREDICTOR PIPELINE COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with exception: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_quick_test():
    """Run a quick test of the pipeline with minimal data."""
    print("🧪 Running quick test of the pipeline...")
    
    try:
        # Import and test data preprocessing
        from data_preprocessing import create_synthetic_dataset
        df = create_synthetic_dataset()
        print(f"✓ Synthetic dataset created: {df.shape}")
        
        # Test preprocessing
        X, y, label_encoder, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("✓ Data preprocessing test passed")
        
        # Test model training (without optimization for speed)
        from model_training import train_model
        model, _, _ = train_model(X_train, y_train, optimize_hyperparams=False)
        print("✓ Model training test passed")
        
        # Test evaluation
        from model_training import evaluate_model_performance
        performance = evaluate_model_performance(model, X_test, y_test, label_encoder)
        print(f"✓ Model evaluation test passed - Accuracy: {performance['accuracy']:.4f}")
        
        print("✅ All tests passed! Pipeline is ready to run.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run quick test
        success = run_quick_test()
    else:
        # Run full pipeline
        success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
