## AI-Based Traffic Congestion Risk Detection

This repository contains an end-to-end machine learning system for detecting
traffic congestion and incident risk using aerial imagery–derived traffic features.
The project includes data analysis, model training, explainability, spatial
visualization, and deployment using Streamlit.

## Solution
  Trained an XGBoost classifier on traffic flow indicators
  Optimized model using ROC AUC
  Performed decision threshold tuning to prioritize incident recall
  Converted probabilities into actionable risk bands (Low / Medium / High)
  Visualized congestion risk using proxy spatial heatmaps
  Explained predictions using SHAP
  Deployed using Streamlit Cloud

## Tech Stack
  Python
  XGBoost
  scikit learn
  SHAP
  Streamlit
  Pandas / NumPy / Matplotlib

## PROJECT STRUCTURE AND NAVIGATION GUIDE

Root Directory
The root directory contains the main application files required to run
and deploy the system.

- app.py
  Main Streamlit application file. This script loads the trained model,
  accepts user inputs, performs congestion risk prediction, and displays
  results through an interactive interface.

- requirements.txt
  Lists all Python dependencies required to run the project locally
  or deploy it on Streamlit Cloud.

- README.txt
  Project overview and navigation guide.


models/
This directory contains trained model artifacts and associated metadata.

- traffic_model.json
  Trained XGBoost model saved using XGBoost’s native format.

- traffic_model_metadata.json
  Stores auxiliary information such as feature order, decision threshold,
  and risk band definitions used during inference.


data/
This directory contains the dataset used for experimentation and analysis.

- dataset.csv
  Tile-level traffic feature dataset provided for the capstone project.
  Each row represents a road segment tile with extracted traffic features
  and a congestion label.


notebooks/
This directory contains exploratory and experimental notebooks.

- experiments.ipynb
  Jupyter notebook used for data exploration, feature analysis,
  model training, evaluation, spatial visualization, and explainability
  (SHAP analysis). Data loading is handled dynamically relative to the
  project root for portability.

## RUNNING THE APPLICATION

To run the application locally:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py

The application will load the trained model and allow congestion risk
prediction using user-provided traffic feature values.

## NOTES

- Spatial visualizations use a proxy grid due to the absence of explicit
  geographic coordinates in the dataset.
- The model is optimized to prioritize congestion recall, supporting
  early detection at the cost of moderate false positives.
- The repository is structured to separate data, models, experiments,
  and deployment code for clarity and reproducibility.