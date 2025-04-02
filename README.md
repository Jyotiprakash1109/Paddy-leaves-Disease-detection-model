# 🌾 Paddy Disease Classifier using TensorFlow

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** model to classify diseases in paddy leaves. The model is built using **TensorFlow & Keras** and is trained on an image dataset containing different types of paddy diseases.

## 🚀 Features
- **Deep Learning-Based Image Classification** using CNN.
- **Automated Training & Evaluation** with data augmentation.
- **Model Optimization** using early stopping, learning rate reduction, and best model checkpointing.
- **Confusion Matrix & Accuracy Reports** for performance evaluation.
- **Single Image Prediction** with confidence scores.
- **Model Saving & Loading** for future inference.

---

## 📂 Project Structure
```
├── dataset/
│   ├── train/   # Training images (divided into subfolders by class)
│   ├── test/    # Test images (divided into subfolders by class)
├── models/
│   ├── paddy_disease_model.keras   # Saved trained model
│   ├── model_metadata.json         # Model metadata (class indices, image size, etc.)
├── src/
│   ├── train.py    # Script for training the model
│   ├── predict.py  # Script for making predictions
│   ├── utils.py    # Helper functions
├── README.md  # Project Documentation
```

---

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/paddy-disease-classifier.git
cd paddy-disease-classifier
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (If Not Pre-Trained)
```bash
python src/train.py --train_dir path/to/train_data --model_dir path/to/save_model
```

### 4️⃣ Evaluate on Test Data
```bash
python src/evaluate.py --test_dir path/to/test_data --model_dir path/to/saved_model
```

### 5️⃣ Predict Disease from an Image
```bash
python src/predict.py --image path/to/image.jpg --model_dir path/to/saved_model
```

---

## 🏗️ Model Architecture
The model consists of **four convolutional blocks**, followed by fully connected layers.

```plaintext
1️⃣ Conv2D (32 filters, 3x3 kernel) → BatchNorm → MaxPooling
2️⃣ Conv2D (64 filters, 3x3 kernel) → BatchNorm → MaxPooling
3️⃣ Conv2D (128 filters, 3x3 kernel) → BatchNorm → MaxPooling
4️⃣ Conv2D (256 filters, 3x3 kernel) → BatchNorm → MaxPooling
🔹 Flatten → Dense(512) → Dropout(0.5) → Output (Softmax)
```

**Optimizer:** Adam (LR = 0.0001)  
**Loss Function:** Categorical Cross-Entropy  
**Metrics:** Accuracy

---

## 📊 Model Performance
After training, the model achieves high accuracy on the test dataset.
- **Confusion Matrix** and **Classification Report** are used for evaluation.
- **Loss & Accuracy Graphs** provide insights into model convergence.

<img src="assets/training_history.png" width="500">

---

## 📷 Sample Prediction Output
**Input Image:**

<img src="assets/sample_leaf.jpg" width="300">

**Model Prediction:**
```
Predicted Disease: Brown Spot
Confidence: 92.5%
```

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit a Pull Request or open an Issue.

