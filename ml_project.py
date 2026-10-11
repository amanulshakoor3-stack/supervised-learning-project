"""
Supervised Learning Project
Implements Linear Regression, Decision Trees, and Random Forest
with training, testing, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def regression_example():
    """Linear Regression example using diabetes dataset"""
    print("="*50)
    print("LINEAR REGRESSION EXAMPLE (Diabetes Dataset)")
    print("="*50)
    
    # Load diabetes dataset
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lr_model, mse, r2

def classification_example():
    """Decision Tree and Random Forest example using breast cancer dataset"""
    print("="*50)
    print("CLASSIFICATION EXAMPLE (Breast Cancer Dataset)")
    print("="*50)
    
    # Load breast cancer dataset
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train models
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Get prediction probabilities for ROC curve
    dt_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate models
    dt_accuracy = accuracy_score(y_test, dt_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_pred, target_names=cancer.target_names))
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=cancer.target_names))
    
    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Decision Tree Confusion Matrix
    dt_cm = confusion_matrix(y_test, dt_pred)
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cancer.target_names, yticklabels=cancer.target_names, ax=axes[0])
    axes[0].set_title('Decision Tree Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Random Forest Confusion Matrix
    rf_cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=cancer.target_names, yticklabels=cancer.target_names, ax=axes[1])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('classification_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize ROC curves
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC for Decision Tree
    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_pred_proba)
    dt_roc_auc = auc(dt_fpr, dt_tpr)
    
    # Calculate ROC for Random Forest
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    rf_roc_auc = auc(rf_fpr, rf_tpr)
    
    plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_roc_auc:.3f})')
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Classification Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dt_model, rf_model, dt_accuracy, rf_accuracy

def main():
    """Main function to run the supervised learning project"""
    print("Supervised Learning Project")
    print("Implementing Linear Regression, Decision Trees, and Random Forest\n")
    
    # Run regression example
    lr_model, mse, r2 = regression_example()
    
    print("\n")
    
    # Run classification example
    dt_model, rf_model, dt_acc, rf_acc = classification_example()
    
    # Summary
    print("="*50)
    print("PROJECT SUMMARY")
    print("="*50)
    print("Algorithms Implemented:")
    print("- Linear Regression (for regression task)")
    print("- Decision Tree Classifier (for classification task)")
    print("- Random Forest Classifier (for classification task)")
    print("\nKey Visualizations Generated:")
    print("- Linear regression: Actual vs Predicted scatter plot")
    print("- Classification: Confusion matrices for both models")
    print("- Classification: ROC curves for both models")
    print("\nFiles saved in current directory:")
    print("- linear_regression_results.png")
    print("- classification_confusion_matrices.png")
    print("- roc_curves.png")
    
if __name__ == "__main__":
    main()