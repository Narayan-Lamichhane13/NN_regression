# Phishing URL Detection with Neural Networks 🛡️🔍

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

This project implements a neural network classification pipeline to detect phishing URLs. Using PyTorch, the pipeline processes the dataset, trains a binary classifier, and evaluates its performance using metrics like accuracy, precision, recall, and the ROC curve.

---

## 🚀 Features
- 🧹 **Data Preprocessing**: Scales features, splits data into train/validation/test sets.
- 🧠 **Neural Network Model**:
  - Multi-layer perceptron with ReLU activation.
  - Dropout layers for regularization.
  - Sigmoid output for binary classification.
- 🔍 **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score.
  - ROC Curve and AUC score.

---

## 📂 Repository Structure
NN_regression/ ├── main.py # Main script orchestrating the pipeline ├── train.py # Training script for the neural network ├── model.py # Defines the neural network architecture ├── evaluate.py # Evaluates the model and visualizes performance ├── data_preparation.py # Handles data loading and preprocessing ├── hyperparameter_tuning.py # Grid search for hyperparameter optimization ├── phishing_dataset.csv # Dataset for training and testing └── README.md # Project documentation

---

## 🧰 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/NN_regression.git
   cd NN_regression
   ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
Place the dataset: Download the Phishing URL Dataset and save it as phishing_dataset.csv in the root directory.
Download Dataset : https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset


🚀 How to Run
Train the Model:
```bash
python main.py
```
Evaluate the Model:
```bash
python evaluate.py
```


## 📊 Training Logs

Below is the training log showing the model's progress over 50 epochs. Both the training loss and validation loss decrease significantly, and the validation accuracy reaches near-perfect levels:

![image](https://github.com/user-attachments/assets/14d2d742-fcef-4af0-b233-c821ee3b384e)


🔧 Future Enhancements
Integrate real-time URL scraping for predictions.
Experiment with CNN or Transformer architectures.
Implement adversarial training to handle sophisticated phishing attempts.

📜 License
This project is licensed under the MIT License.
```bash

---

### **How to Use**
1. Copy the content above.
2. Paste it into your `README.md` file.
3. Replace placeholders like `your-username` with your actual GitHub username. 

This format is simple yet visually appealing, with clear headings and sections!

```
