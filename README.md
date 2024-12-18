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
```
/rope-damage-sensors
│── src/                        # Source code
│   ├── train.py                # Model training script
│   ├── retrain.py              # Model retraining script
│   └── helper.py               # Helper functions
│── data/                       # Data files
│   ├── train                   # Training data
│       ├── Fold_1              # Training data is split to 4-fold cross-validation data, i.e., 4 folds
│           ├── Train           # Each fold contains Train and Validation folders
│               ├── 0           # Class name for Normal ropes
│                   ├── images
│               ├── 1           # Class name for Damaged ropes
│                   ├── images
│           ├── Validation
│               ├── 0
│               ├── 1
│       ...
│       ├── Fold_4
│   ├── test                   # Testing data
│   └── combined_train         # retrain.py combines train and validation data into this folder
│── results/                   # Training results and logs
│── LICENSE                    # License file
│── README.md                  # Project description
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignored files and folders
```

## **⚙️ Installation**
**1.	Clone the repository:**
```
git clone https://github.com/TuomasJalonen/rope-damage-sensors.git
cd rope-damage-sensors
```
**2.	Create a virtual environment:**
```
python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows
```
**3.	Install dependencies:**
```
pip install -r requirements.txt
```
## **🧠 Usage**
1.	Prepare your dataset: Follow the folder structure described above. If you want to use the dataset described in the paper, you can request it from roope.mellanen@konecranes.com.
2.	Train and evaluate the model using 4-fold cross-validation:
```
python src/train.py --model MODEL_NAME
python src/train.py --model CNN9 --epochs 150 --batch_size 32 # Example, you can change the model, epochs and batch size.
```
3. Retrain and evaluate the model so that validation data is included in the training set
```
python src/retrain.py --model MODEL_NAME
python src/retrain.py --model CNN9 --epochs 150 --batch_size 32 # Example, you can change the model, epochs and batch size.
```
## **🔬 Models Available**
MODEL_NAME can be one of the following. You can easily add or modify them in train.py and retrain.py.
1. Our models:
```
CNN1 CNN2 CNN3 CNN4 CNN5 CNN6 CNN7 CNN8 CNN9 CNN10 CNN11 CNN12 CNN13 CNN14 CNN15 CNN16
```
2. Zhou et al. (2019) models:
```
Zhou_2019_1 Zhou_2019_2 Zhou_2019_3 Zhou_2019_4 Zhou_2019_5 Zhou_2019_6
```
3. Zhou et al. (2021) models:
```
Zhou_2021_1 Zhou_2021_2 Zhou_2021_3 Zhou_2021_4 Zhou_2021_5 Zhou_2021_6 Zhou_2021_7
```
4. Schuler at al. (2022) models:
```
kDenseNet_BC_L100_12ch_1 kDenseNet_BC_L100_12ch_2 kDenseNet_BC_L100_12ch_3 kDenseNet_BC_L100_12ch_4
```
## **📊 Results**
After training, check the results/ folder for:
	•	Training logs
	•	Evaluation metrics
	•	Model predictions

## **📜 License**
This project is licensed under the MIT License - see the LICENSE file for details.

## **🤝 Contributing**
Feel free to submit issues, fork the repo, and submit pull requests.

## **👤 Author**
Tuomas Jalonen

## 📚 **Citation Information**
