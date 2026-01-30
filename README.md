# AI Based Traffic Congestion Risk Detection 🚦

End to end machine learning system for detecting traffic congestion and incident risk using aerial imagery derived traffic features.

## 🔍 Problem Overview
Traditional traffic monitoring systems rely on fixed thresholds and manual observation. This project uses machine learning to estimate congestion *risk* probabilistically, enabling early detection and spatial monitoring.

## 🧠 Solution
  Trained an XGBoost classifier on traffic flow indicators
  Optimized model using ROC AUC
  Performed decision threshold tuning to prioritize incident recall
  Converted probabilities into actionable risk bands (Low / Medium / High)
  Visualized congestion risk using proxy spatial heatmaps
  Explained predictions using SHAP
  Deployed using Streamlit Cloud

## ⚙️ Tech Stack
  Python
  XGBoost
  scikit learn
  SHAP
  Streamlit
  Pandas / NumPy / Matplotlib

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
