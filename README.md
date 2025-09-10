# 🔍 ICS Anomaly Detection with Lightweight Transformers

> A dissertation project for the MSc Advanced Computer Science at the **University of Birmingham**.  
> This project explores the use of **self-attention (Transformers)** for **real-time anomaly detection** in **Industrial Control Systems (ICS)**, demonstrating superior accuracy and latency compared to traditional models such as **LSTMs** and **Autoencoders**.

---

## 📖 Project Overview
Industrial Control Systems (ICS) secure critical infrastructure like **water treatment plants, power grids, and manufacturing systems**.  
They are increasingly vulnerable to cyberattacks due to IT/OT convergence. Detecting anomalies in real-time is vital to protect these systems.

This project introduces a **Lightweight Transformer-based model** tailored for anomaly detection in ICS, benchmarked against LSTM and LSTM Autoencoder baselines, using the **SWaT dataset**.

---

## 🚀 Key Achievements
- ✅ **Lightweight Transformer model** designed for ICS anomaly detection.  
- ✅ **Higher F1-Score (0.80)** compared to LSTM Classifier (0.70) and LSTM Autoencoder (0.65).  
- ✅ **Ultra-low inference latency** of **0.87 ms** per sample (well below the 100 ms real-time threshold).  
- ✅ End-to-end pipeline: **preprocessing → training → evaluation → latency testing**.  
- ✅ Case study: **Pump manipulation attack** detection, showing how attention links sensor correlations.  
- ✅ Includes **visualizations** for architectures, results, and attention heatmaps.

---

## 📊 Results Summary

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| **Transformer**    | 0.9778   | 0.8473    | 0.7644 | **0.8037** |
| **LSTM Classifier**| 0.9728   | 0.9919    | 0.5467 | 0.7049 |
| **LSTM Autoencoder**| 0.9680  | 0.9262    | 0.5022 | 0.6513 |

⏱️ **Latency (Transformer): 0.87 ms** on CPU (edge-device simulation).  

---

## 🖼️ Visual Insights

### Model Comparison Pipeline
<img width="1000" height="600" alt="model_performance_comparison" src="https://github.com/user-attachments/assets/4b9bf6cf-7236-4d89-906c-fd85fa63c1a7" />


### Training & Evaluation Workflow
<img width="960" height="1025" alt="anomaly_transformer_flowchart" src="https://github.com/user-attachments/assets/295ebacd-fdf6-4471-b827-f7631a9449b7" />


### Case Study: Pump Manipulation Attack
<img width="1000" height="600" alt="ChatGPT Image Aug 19, 2025, 04_32_16 PM" src="https://github.com/user-attachments/assets/161dca46-4b3f-471d-85cc-6b7970aee078" />


### Transformer Attention Heatmap
<img width="1000" height="900" alt="attention_heatmap" src="https://github.com/user-attachments/assets/03ddbd66-55c6-45f3-a7e8-55d200dbc800" />


---

## ⚙️ Tech Stack
- **Languages & Frameworks**: Python, PyTorch, NumPy, Pandas, Scikit-learn  
- **Data**: [SWaT Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) (Secure Water Treatment testbed)  
- **ML Techniques**: Transformers, LSTMs, Autoencoders, SMOTE (for class imbalance)  
- **Tools**: Matplotlib, Seaborn (visualization), Git, LaTeX  

---
## 📂 Repository Structure
```bash
ics-anomaly-detection-transformer/
│── data/                # Processed datasets (SWaT/WADI)
│── models/              # Saved trained models
│── report/              # Dissertation & visualizations
│── src/                 # Core source code
│   │── model.py         # Transformer architecture
│   │── train.py         # Transformer training pipeline
│   │── train_baseline.py# LSTM & Autoencoder training
│   │── evaluate.py      # Evaluation scripts (metrics, confusion matrices)
│   │── measure_latency.py # Latency benchmarking
│   │── preprocess.py    # Data preprocessing (windowing, cleaning)
│   │── baseline_models.py # LSTM Classifier & Autoencoder
│── requirements.txt     # Dependencies
│── main.py              # Entry point for training & evaluation
│── README.md            # Project documentation



⭐ If you found this project useful, consider giving it a star !
