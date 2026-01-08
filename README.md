# Face Mask Detection using TensorFlow & MobileNetV2

---

## Project Overview

The goal of this project is to build a robust image classification model capable of detecting whether a person is wearing a face mask or not. This type of system can be applied in:

* Public safety monitoring
* Access control systems
* Compliance analytics
* Computer vision learning demonstrations

Rather than relying on a Kaggle runtime, the entire pipeline is implemented and executed **locally**, making it reproducible and deployment-ready.

---

## Problem Definition

**Task:** Binary Image Classification
**Classes:**

* `with_mask`
* `without_mask`

**Input:** RGB images of human faces
**Output:** Probability and class label indicating mask usage

---

## Dataset Structure

The dataset is organized in a folder-based classification format:

```text
Face Mask detection/
│
├── data/
│   ├── with_mask/        # Images of people wearing face masks
│   └── without_mask/     # Images of people not wearing face masks
│
├── eval/                 # Optional unseen images (not used in training)
└── facemask_mobilenetv2.h5
```

* The `data/` directory is used for training and validation
* The `eval/` directory is reserved for optional inference/testing
* Training and validation split is performed programmatically

---

## Tech Stack

* **Programming Language:** Python 3.11
* **Deep Learning Framework:** TensorFlow / Keras
* **Model Architecture:** MobileNetV2 (Transfer Learning)
* **Libraries:**

  * NumPy
  * Matplotlib
  * Scikit-learn
  * Pillow

---

## Model Architecture

* **Base Model:** MobileNetV2 (pretrained on ImageNet)
* **Strategy:** Transfer Learning
* **Custom Head:**

  * Global Average Pooling
  * Dense layer (ReLU)
  * Dropout (regularization)
  * Sigmoid output for binary classification

The base model was frozen during training to prevent overfitting and leverage pretrained feature extraction.

---

## Data Preprocessing & Augmentation

To improve generalization, the following preprocessing steps were applied:

* Image resizing to `224 × 224`
* Pixel normalization (`rescale=1./255`)
* Data augmentation:

  * Rotation
  * Zoom
  * Width & height shifts
  * Horizontal flipping

A **20% validation split** was used.

---

## Training Details

* **Loss Function:** Binary Cross-Entropy
* **Optimizer:** Adam (learning rate = 0.0001)
* **Batch Size:** 32
* **Epochs:** 10
* **Training Strategy:** Mini-batch gradient descent with augmentation

Training showed stable convergence with no signs of overfitting.

---

## Model Evaluation

### Classification Report

```
              precision    recall  f1-score   support

   with_mask       0.97      0.98      0.97      1921
without_mask       0.98      0.97      0.97      1947

    accuracy                           0.97      3868
   macro avg       0.97      0.97      0.97      3868
weighted avg       0.97      0.97      0.97      3868
```

### Confusion Matrix Summary

* **True Positives (with_mask):** 1,882
* **True Negatives (without_mask):** 1,884
* **False Positives:** 63
* **False Negatives:** 39

This indicates strong generalization with minimal misclassification and balanced performance across classes.

---

## Training Behavior Analysis

* Training accuracy and loss showed batch-level volatility due to data augmentation
* Validation accuracy remained stable between **96–98%**
* Validation loss decreased consistently, confirming robust learning

This behavior is expected and desirable in transfer learning setups.

---

## Model Saving

The trained model is saved as:

```bash
facemask_mobilenetv2.h5
```

This file can be directly loaded for inference or deployment.

---

## How to Run the Project

### Install dependencies

```bash
pip install tensorflow scikit-learn numpy matplotlib pillow
```

### Train the model

Run the training script or notebook cells responsible for:

* Data loading
* Model building
* Training

### Evaluate the model

Evaluation includes:

* Confusion matrix
* Precision, recall, F1-score

### Inference (optional)

Use the saved model to predict on new images or the `eval/` folder.

---

## Future Improvements

* Fine-tune the last layers of MobileNetV2
* Add real-time webcam detection
* Deploy using Streamlit or FastAPI
* Convert to TensorFlow Lite for mobile/edge deployment

---

## Author

**David Obi**
Data Scientist | Machine Learning Engineer

---

## License

This project is for educational and research purposes. You are free to modify and extend it with proper attribution.

---

*If you found this project useful, feel free to star the repository!*
