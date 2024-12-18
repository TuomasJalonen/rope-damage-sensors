# **Real-Time Damage Detection in Fiber Lifting Ropes Using Lightweight Convolutional Neural Networks**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## **📚 Description**
This project focuses on automating damage detection in fiber lifting ropes using lightweight convolutional neural networks (CNNs). Damage in crane lifting ropes presents significant health and safety risks, making regular inspections essential. However, manual inspections are time-consuming, prone to human error, and interrupt operational workflow. This project proposes a deep learning-based approach to streamline the inspection process through real-time detection using computer vision.

### **Key Objectives:**
- Develop a vision-based system for detecting damage in synthetic fiber rope images.
- Create a camera-based apparatus to capture the surface of the lifting rope during operation.
- Use expert-annotated images indicating rope conditions ("normal" or "damaged") for model training.
- Design and evaluate lightweight CNN models optimized for real-time damage detection.

### **Core Contributions:**
- **Model Development:** Lightweight CNNs were systematically designed and evaluated.
- **Performance Metrics:** Models were benchmarked with high detection accuracy, precision, recall, F1-score, and AUC.
- **Real-Time Capability:** The system demonstrated real-time performance and a low memory footprint.
- **Robustness:** The solution proved resilient across different operational and environmental conditions.
- **Industrial Relevance:** The proposed model is suitable for deployment in industries such as lifting, mooring, towing, climbing, and sailing.

### **Experimental Results:**
- **Accuracy:** 96.5%
- **Precision:** 94.8%
- **Recall:** 98.3%
- **F1-Score:** 96.5%
- **AUC:** 99.3%

This project leverages deep learning and computer vision techniques to improve operational safety, reduce inspection times, and minimize downtime, enabling continuous and automated rope damage monitoring.

---

## **📂 Folder Structure**
```plaintext
/your_project
│── src/                  # Source code
│   ├── train.py          # Model training script
│   ├── retrain.py        # Model retraining script
│   └── helper.py         # Helper functions

│── data/                 # Data files
│── results/              # Training results and logs
│── models/               # Saved models
│── LICENSE               # License file
│── README.md             # Project description
│── requirements.txt      # Python dependencies
│── .gitignore            # Ignored files and folders
