# Supervised Learning Project

This project demonstrates the implementation of supervised learning algorithms including Linear Regression, Decision Trees, and Random Forest. It covers model training, testing, and performance visualization.

## Project Structure

- `ml_project.py`: Main Python script implementing the algorithms and generating visualizations
- `linear_regression_results.png`: Visualization of Linear Regression results (Actual vs Predicted)
- `classification_confusion_matrices.png`: Confusion matrices for Decision Tree and Random Forest classifiers
- `roc_curves.png`: ROC curves for both classification models

## Algorithms Implemented

1. **Linear Regression** - Applied to the diabetes dataset for regression task
2. **Decision Tree Classifier** - Applied to the breast cancer dataset for classification task
3. **Random Forest Classifier** - Applied to the breast cancer dataset for classification task

## Key Features

- Data loading and preprocessing using scikit-learn's built-in datasets
- Train/test split (80/20) for model evaluation
- Model training and prediction
- Performance metrics calculation:
  - For regression: Mean Squared Error (MSE), R² Score
  - For classification: Accuracy, Precision, Recall, F1-score
- Visualization generation:
  - Regression: Actual vs Predicted scatter plot
  - Classification: Confusion matrices (heatmaps)
  - Classification: ROC curves with AUC scores

## Requirements

- Python 3.x
- scikit-learn
- matplotlib
- pandas
- numpy
- seaborn

## Installation

1. Clone or download this repository
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   ```
3. Install required packages:
   ```bash
   pip install scikit-learn matplotlib pandas numpy seaborn
   ```

## Usage

Run the main script:
```bash
python ml_project.py
```

The script will:
1. Execute the Linear Regression example on the diabetes dataset
2. Execute the Decision Tree and Random Forest examples on the breast cancer dataset
3. Generate and save visualization plots as PNG files
4. Print performance metrics to the console

## Results

### Linear Regression (Diabetes Dataset)
- Mean Squared Error: ~2900.19
- R² Score: ~0.45

### Classification (Breast Cancer Dataset)
- Decision Tree Accuracy: ~94.74%
- Random Forest Accuracy: ~96.49%

## Visualizations

1. **Linear Regression Results**: Shows the correlation between actual and predicted values
2. **Confusion Matrices**: Displays true positives, false positives, true negatives, and false negatives for both classifiers
3. **ROC Curves**: Illustrates the trade-off between sensitivity and specificity for various threshold settings

## Notes

- The project uses scikit-learn's built-in datasets for simplicity and reproducibility
- Random forest uses 100 estimators
- All visualizations are saved as high-resolution PNG files (300 DPI)
- The script includes error handling and suppresses warnings for cleaner output

## License

This project is for educational purposes and can be freely used and modified.