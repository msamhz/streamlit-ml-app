"""
Breast Cancer Prediction App using Streamlit and RandomForestClassifier.
"""

import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

selected_features = [
    'worst perimeter', 'mean concave points', 'worst radius',
    'worst concave points', 'worst area', 'mean concavity'
]

@st.cache_data
def load_data():
    """
    Load and preprocess the breast cancer dataset.
    """
    data = load_breast_cancer()
    df_data = pd.DataFrame(data.data, columns=data.feature_names)
    df_data = df_data[selected_features]
    df_data['cancer'] = data.target
    return df_data, data.target_names

def get_min(dataframe):
    """
    Get the minimum values for the selected features.
    """
    return dict(dataframe.describe()[selected_features].min())

def get_max(dataframe):
    """
    Get the maximum values for the selected features.
    """
    return dict(dataframe.describe()[selected_features].max())

df, target_names = load_data()
model = RandomForestClassifier()
model.fit(df.drop('cancer', axis=1), df['cancer'])

st.sidebar.title('Input Parameters')
worst_peri = st.sidebar.slider(
    'Worst Perimeter', get_min(df)['worst perimeter'], get_max(df)['worst perimeter']
)
mean_conc = st.sidebar.slider(
    'Mean Concave Points', get_min(df)['mean concave points'], get_max(df)['mean concave points']
)
worst_rad = st.sidebar.slider(
    'Worst Radius', get_min(df)['worst radius'], get_max(df)['worst radius']
)
worst_conc = st.sidebar.slider(
    'Worst Concave Points', get_min(df)['worst concave points'], get_max(df)['worst concave points']
)
worst_area = st.sidebar.slider(
    'Worst Area', get_min(df)['worst area'], get_max(df)['worst area']
)
mean_concavity = st.sidebar.slider(
    'Mean Concavity', get_min(df)['mean concavity'], get_max(df)['mean concavity']
)

input_data = [[
    worst_peri, mean_conc, worst_rad, worst_conc, worst_area, mean_concavity
]]

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prediction_classification = target_names[prediction]
    st.write('Prediction: ', prediction_classification)

st.write('Breast Cancer Classification')