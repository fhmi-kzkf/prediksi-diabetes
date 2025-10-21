import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('diabetes.csv')

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# Data preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize models with basic parameters
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

# Define hyperparameter grids for tuning
param_grids = {
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}

# Train and evaluate models with hyperparameter tuning
results = {}
trained_models = {}
best_params = {}

print("\n" + "="*50)
print("MODEL TRAINING AND HYPERPARAMETER TUNING")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for models that benefit from it
    X_train_data = X_train_scaled if name in ['K-Nearest Neighbors', 'Logistic Regression', 'Support Vector Machine'] else X_train
    X_test_data = X_test_scaled if name in ['K-Nearest Neighbors', 'Logistic Regression', 'Support Vector Machine'] else X_test
    
    # Perform hyperparameter tuning for selected models
    if name in param_grids:
        print(f"Tuning hyperparameters for {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_data, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        # For models without tuning (like Naive Bayes), train normally
        best_model = model
        best_model.fit(X_train_data, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test_data)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    trained_models[name] = best_model
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Find the best model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
best_accuracy = results[best_model_name]

print("\n" + "="*50)
print("MODEL COMPARISON RESULTS")
print("="*50)
print(f"Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Display best parameters for the best model
if best_model_name in best_params:
    print(f"Best parameters for {best_model_name}: {best_params[best_model_name]}")

# Sort models by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nModel Rankings:")
for i, (name, accuracy) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {accuracy:.4f}")

# Visualization
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Diabetes Prediction Model Comparison (Tuned Models)', fontsize=16)

# 1. Model Accuracy Comparison
axes[0, 0].bar(range(len(results)), list(results.values()), color=['skyblue' if name != best_model_name else 'orange' for name in results.keys()])
axes[0, 0].set_xticks(range(len(results)))
axes[0, 0].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylim(0, 1)

# Add value labels on bars
for i, (name, accuracy) in enumerate(results.items()):
    axes[0, 0].text(i, accuracy + 0.01, f'{accuracy:.3f}', ha='center', va='bottom')

# 2. Confusion Matrix for Best Model
X_test_data = X_test_scaled if best_model_name in ['K-Nearest Neighbors', 'Logistic Regression', 'Support Vector Machine'] else X_test
y_pred_best = best_model.predict(X_test_data)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    axes[0, 2].barh(range(len(feature_importance)), feature_importance['importance'], color='lightcoral')
    axes[0, 2].set_yticks(range(len(feature_importance)))
    axes[0, 2].set_yticklabels(feature_importance['feature'])
    axes[0, 2].set_xlabel('Importance')
    axes[0, 2].set_title('Feature Importance')
else:
    axes[0, 2].text(0.5, 0.5, f'{best_model_name}\ndoes not provide\nfeature importance', 
                    ha='center', va='center', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Feature Importance')

# 4. Confusion Matrices for All Models
model_names = list(results.keys())
for i, name in enumerate(model_names[:2]):  # Show first two models
    X_test_data = X_test_scaled if name in ['K-Nearest Neighbors', 'Logistic Regression', 'Support Vector Machine'] else X_test
    y_pred = trained_models[name].predict(X_test_data)
    
    cm = confusion_matrix(y_test, y_pred)
    row, col = (1, i) if i == 0 else (1, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, i])
    axes[1, i].set_title(f'Confusion Matrix - {name}')
    axes[1, i].set_xlabel('Predicted')
    axes[1, i].set_ylabel('Actual')

# 5. Data Distribution
axes[1, 2].pie(df['Outcome'].value_counts(), labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
axes[1, 2].set_title('Distribution of Diabetes Outcomes')

plt.tight_layout()
plt.savefig('model_comparison_tuned.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the best model and scaler
with open('best_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nBest tuned model ({best_model_name}) and scaler saved successfully!")

# Print feature names for reference
print("\nFeature names:")
for i, feature in enumerate(X.columns):
    print(f"{i+1}. {feature}")