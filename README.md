# Breast Cancer Prediction with Streamlit

This project is a web application for predicting breast cancer using a RandomForestClassifier. The application is built with Streamlit and allows users to input various features to get a prediction on whether the cancer is malignant or benign.

## Features

- **User Input**: Users can input values for specific features related to breast cancer.
- **Prediction**: The app predicts whether the cancer is malignant or benign based on the input features.
- **Visualization**: Simple and interactive UI built with Streamlit.

## Selected Features

The following features are used for prediction:
- Worst Perimeter
- Mean Concave Points
- Worst Radius
- Worst Concave Points
- Worst Area
- Mean Concavity

## How to Run

1. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the application**:
    ```sh
    streamlit run app.py
    ```

## Usage

- Open the web application in your browser.
- Use the sliders in the sidebar to input values for the features.
- Press the "Predict" button to get the prediction.

## Dependencies

- pandas
- scikit-learn
- streamlit