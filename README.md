# Diabetes Prediction Application

This project predicts whether a patient has diabetes based on medical data using machine learning algorithms.

## Project Structure

- `diabetes.csv`: The dataset containing patient information
- `diabetes_prediction.py`: Script to train and compare machine learning models
- `diabetes_app.py`: Streamlit web application for making predictions
- `best_model.pkl`: The trained machine learning model (created after running training)
- `scaler.pkl`: The feature scaler (created after running training)
- `model_comparison.png`: Visualization of model performance (created after running training)
- `requirements.txt`: List of required Python packages

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Models

First, run the training script to train multiple machine learning models and select the best one:

```bash
python diabetes_prediction.py
```

This will:
- Train 5 different machine learning models
- Compare their performance
- Save the best model as `best_model.pkl`
- Save the feature scaler as `scaler.pkl`
- Generate performance visualizations in `model_comparison.png`

### 2. Run the Streamlit Application

After training, run the Streamlit app:

```bash
streamlit run diabetes_app.py
```

This will open a web browser with the application where you can:
- Input patient medical data using manual entry or sliders
- Get diabetes risk predictions with detailed probability analysis
- View model performance visualizations
- Access information about the project and features

## Enhanced Features

The Streamlit application now includes:

1. **Interactive Input Methods**:
   - Manual number input for precise values
   - Sliders for intuitive value adjustment
   - Tabbed interface for different input methods

2. **Enhanced Visualization**:
   - Color-coded result display
   - Probability distribution charts
   - Risk level indicators
   - Metric cards for key statistics

3. **Navigation System**:
   - Prediction mode for making new predictions
   - Data Visualization mode to view model performance
   - About section with detailed information

4. **User Experience Improvements**:
   - Custom styling with CSS
   - Loading animations
   - Detailed result explanations
   - Risk-based recommendations
   - Responsive layout for different screen sizes

## Features

The application uses the following medical features for prediction:
1. Number of Pregnancies
2. Glucose Level (mg/dL)
3. Blood Pressure (mm Hg)
4. Skin Thickness (mm)
5. Insulin Level (mu U/ml)
6. BMI (Body Mass Index)
7. Diabetes Pedigree Function
8. Age (years)

## Model Performance

The models are compared based on accuracy. The best performing model is automatically selected and used in the Streamlit application:

- Random Forest (Accuracy: ~76%)
- Support Vector Machine (Accuracy: ~75%)
- Logistic Regression (Accuracy: ~71%)
- Naive Bayes (Accuracy: ~71%)
- K-Nearest Neighbors (Accuracy: ~70%)

## Dataset

The dataset used is the Pima Indians Diabetes Database which contains data on female patients at least 21 years old of Pima Indian heritage.

## License

This project is for educational purposes only.