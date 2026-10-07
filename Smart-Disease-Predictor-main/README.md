# 🏥 Smart Disease Predictor

A complete end-to-end machine learning project that predicts diseases based on user-entered symptoms using Random Forest classification with an interactive Streamlit web interface.

## 🎯 Project Overview

This project implements a comprehensive disease prediction system that:
- Uses a synthetic dataset with 15 diseases and 40+ symptoms
- Trains a Random Forest classifier with hyperparameter optimization
- Achieves high accuracy through cross-validation and feature engineering
- Provides an interactive web interface for real-time predictions
- Includes comprehensive evaluation metrics and visualizations

## 📊 Key Features

- **Smart Prediction**: AI-powered disease prediction from symptom input
- **High Accuracy**: Optimized Random Forest model targeting ≥90% accuracy
- **Interactive UI**: Modern Streamlit interface with real-time predictions
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and ROC curves
- **Feature Analysis**: Importance ranking of symptoms for model decisions
- **Medical Disclaimer**: Clear warnings about educational use only

## 🏗️ Project Structure

```
smart-disease-predictor/
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # Model training and optimization
│   ├── model_evaluation.py       # Evaluation metrics and visualization
│   └── app.py                    # Streamlit web application
├── data/                         # Dataset storage
├── models/                       # Saved models and encoders
├── results/                      # Evaluation reports and plots
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd smart-disease-predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Execute the full ML pipeline
python main.py
```

This will:
- Create and preprocess the dataset
- Train the Random Forest model with hyperparameter optimization
- Evaluate the model and generate comprehensive reports
- Save all model components for the web app

### 3. Launch the Web Application

```bash
# Start the Streamlit app
streamlit run src/app.py
```

Open your browser to `http://localhost:8501` to access the interactive interface.

### 4. Quick Test (Optional)

```bash
# Run a quick test of the pipeline
python main.py --test
```

## 📋 Requirements

### Python Dependencies
- `scikit-learn==1.3.0` - Machine learning library
- `pandas==2.0.3` - Data manipulation
- `numpy==1.24.3` - Numerical computing
- `matplotlib==3.7.2` - Plotting and visualization
- `seaborn==0.12.2` - Statistical visualization
- `streamlit==1.28.1` - Web application framework
- `imbalanced-learn==0.11.0` - Handling imbalanced datasets
- `plotly` - Interactive visualizations (for Streamlit app)

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Web browser for Streamlit interface

## 🔬 Technical Details

### Dataset
- **Type**: Synthetic disease-symptom dataset
- **Diseases**: 15 different diseases (Common Cold, Flu, Migraine, Diabetes, etc.)
- **Symptoms**: 40+ unique symptoms
- **Samples**: 750 total samples (50 per disease)
- **Format**: Binary features (symptom present/absent)

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Optimization**: RandomizedSearchCV with 5-fold cross-validation
- **Parameters Tuned**:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2', None]

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages
- **Confusion Matrix**: Detailed class-wise performance
- **ROC Curves**: Multi-class ROC analysis

## 🎮 Usage Guide

### Web Application

1. **Select Symptoms**: Choose from the dropdown menu of available symptoms
2. **Get Prediction**: Click "Predict Disease" to analyze your symptoms
3. **View Results**: See the most likely disease with confidence scores
4. **Explore Analysis**: Check feature importance and probability distributions

### Programmatic Usage

```python
from src.model_training import load_model, predict_disease

# Load trained model
model, label_encoder, feature_names, _ = load_model()

# Make prediction
symptoms = ['fever', 'cough', 'fatigue']
predicted_disease, probabilities = predict_disease(
    model, label_encoder, feature_names, symptoms
)

print(f"Predicted disease: {predicted_disease}")
print(f"Top 3 probabilities: {list(probabilities.items())[:3]}")
```

## 📈 Expected Performance

- **Target Accuracy**: ≥90%
- **Training Time**: 2-5 minutes (depending on hardware)
- **Prediction Time**: <1 second per prediction
- **Model Size**: ~2-5 MB (saved as pickle file)

## 🔧 Customization

### Adding New Diseases
1. Modify the `create_synthetic_dataset()` function in `src/data_preprocessing.py`
2. Add new disease-symptom mappings
3. Retrain the model with `python main.py`

### Adding New Symptoms
1. Update the symptom lists in the dataset creation function
2. Ensure feature names are consistent across the pipeline
3. Retrain the model

### Changing the Model
1. Modify the `train_model()` function in `src/model_training.py`
2. Update hyperparameter grids as needed
3. Adjust evaluation metrics if necessary

## 🚨 Important Disclaimers

### Medical Disclaimer
⚠️ **This tool is for educational and research purposes only!**

- **NOT a substitute** for professional medical advice
- **Always consult** with qualified healthcare providers
- **Emergency situations** require immediate professional attention
- **Predictions may not be 100% accurate** - use with caution

### Technical Limitations
- Model performance depends on training data quality
- Synthetic dataset may not reflect real-world symptom patterns
- Binary symptom encoding may oversimplify symptom severity
- No temporal or progression information considered

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   # Ensure you're in the project directory
   cd smart-disease-predictor
   pip install -r requirements.txt
   ```

2. **Model files not found**
   ```bash
   # Run the training pipeline first
   python main.py
   ```

3. **Streamlit app won't start**
   ```bash
   # Check if Streamlit is installed
   pip install streamlit
   streamlit run src/app.py
   ```

4. **Memory issues during training**
   - Reduce dataset size in `create_synthetic_dataset()`
   - Use fewer hyperparameter combinations
   - Close other applications to free memory

### Getting Help
- Check the console output for detailed error messages
- Ensure all dependencies are installed correctly
- Verify Python version compatibility (3.8+)

## 📊 Output Files

After running the pipeline, you'll find:

### Models Directory (`models/`)
- `disease_predictor.pkl` - Trained Random Forest model
- `label_encoder.pkl` - Disease label encoder
- `feature_names.pkl` - List of symptom features
- `performance_metrics.pkl` - Model performance metrics

### Results Directory (`results/`)
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curves.png` - ROC curves for all diseases
- `class_distribution.png` - Dataset class distribution
- `evaluation_metrics.pkl` - Detailed evaluation metrics
- `evaluation_summary.csv` - Summary metrics in CSV format

### Data Directory (`data/`)
- `synthetic_disease_data.csv` - Generated dataset

## 🔮 Future Enhancements

- [ ] Integration with real medical datasets
- [ ] Support for symptom severity levels
- [ ] Temporal symptom progression modeling
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] API endpoint for external integration
- [ ] Advanced visualization dashboard
- [ ] Model explainability features

## 📝 License

This project is created for educational purposes. Please ensure proper attribution if used in academic or commercial contexts.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Improve documentation
- Add new disease-symptom mappings
- Enhance the user interface

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**

*Last updated: 2024*
