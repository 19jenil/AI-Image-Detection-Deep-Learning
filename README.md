# AI Image Detection Using Deep Learning

A deep learning-based web application for detecting whether an image is **Real** or **AI-Generated**. The project compares multiple CNN and transfer-learning architectures and integrates the best-performing model into a Django web application.

## 📌 Project Overview

The rapid development of generative AI has made it increasingly difficult to distinguish AI-generated images from real images. This project investigates deep learning approaches for binary image classification and develops a web-based detection system.

The project follows a complete machine learning workflow:

**Dataset Acquisition → Data Cleaning → Duplicate Detection → Leakage-Aware Splitting → Preprocessing → Augmentation → Model Training → Fine-Tuning → Evaluation → Django Deployment**

## 🎯 Objectives

- Detect AI-generated images using deep learning.
- Investigate and reduce potential data leakage.
- Compare different CNN and transfer-learning architectures.
- Evaluate models using multiple performance metrics.
- Deploy the best-performing model through a Django web application.
- Provide users with prediction results and prediction history.

## 🗂️ Dataset

The project uses the **CIFAKE (Real and AI-Generated Synthetic Images)** dataset.

Initial dataset:

- **120,000 images**
- 60,000 Real
- 60,000 AI-Generated
- Image size: 32 × 32
- RGB images

Duplicate and leakage analysis was performed before model training.

After cleaning and removing duplicate-related leakage:

- **118,664 images**
- Training: 78,961
- Validation: 19,741
- Test: 19,962

Exact hash overlap between the final train, validation and test splits was verified to be zero.

## 🔬 Methodology

### 1. Data Cleaning

The dataset was inspected for:

- Corrupted images
- Duplicate images
- Image dimensions and colour modes
- Class distribution
- Potential train-test leakage

MD5 hashing was used to identify exact duplicate images.

### 2. Preprocessing

Images were:

- Converted to RGB
- Resized to **224 × 224**
- Converted to `float32`
- Normalized to the range **0–1**

### 3. Data Augmentation

Training images were augmented using:

- Random horizontal flipping
- Random rotation
- Random translation
- Random zoom

This was used to reduce overfitting and improve model generalisation.

### 4. TensorFlow Data Pipeline

The TensorFlow pipeline performs:

```text
Load Image
    ↓
Decode JPEG
    ↓
Convert to RGB
    ↓
Resize to 224 × 224
    ↓
Normalize to 0–1
    ↓
Apply Augmentation
    ↓
Batch (32)
    ↓
Prefetch
    ↓
Model
```

## 🧠 Models

Four architectures were investigated:

### Baseline CNN

A custom CNN was developed as a baseline reference model.

Architecture includes:

- Conv2D layers
- MaxPooling
- GlobalAveragePooling
- Dense layer
- Dropout
- Sigmoid output

### ResNet50

ResNet50 with ImageNet pretrained weights was used for transfer learning.

A two-stage training strategy was applied:

**Stage 1**
- Freeze the pretrained backbone
- Train the classification head

**Stage 2**
- Unfreeze deeper layers
- Fine-tune using a low learning rate

### MobileNetV2

MobileNetV2 was included as a lightweight transfer-learning alternative with significantly fewer parameters than ResNet50.

### EfficientNetB0

EfficientNetB0 was evaluated as another efficient transfer-learning architecture. In the tested configuration, it remained close to chance-level performance and therefore was not selected for deployment.

## 📊 Results

| Model | Accuracy | ROC-AUC | Result |
|---|---:|---:|---|
| Baseline CNN | 95.01% | 98.80% | Baseline |
| ResNet50 Stage 1 | 96.78% | 99.52% | Improved |
| **ResNet50 Fine-Tuned** | **98.17%** | **99.76%** | **Selected** |
| MobileNetV2 Stage 1 | 91.41% | 97.39% | Lower |
| MobileNetV2 Fine-Tuned | 96.98% | 99.60% | Lightweight alternative |
| EfficientNetB0 Stage 1 | ~50.66% | 50.00% | Not selected |

### Selected Model

**Fine-tuned ResNet50** achieved the best observed validation performance:

- Accuracy: **98.17%**
- Precision: **98.14%**
- Recall: **98.14%**
- F1 Score: **98.14%**
- ROC-AUC: **99.76%**

The fine-tuned ResNet50 model was therefore selected for integration into the Django application.

## 🌐 Web Application

The trained models are integrated into a Django web application.

### Main Features

- User registration
- User login/logout
- Image upload
- AI-generated image detection
- Multiple model selection
- Confidence score
- Prediction history
- Model comparison
- About/project information

### Detection Workflow

```text
User Uploads Image
        ↓
Select Detection Model
        ↓
Django Backend
        ↓
Image Preprocessing
        ↓
Trained CNN Model
        ↓
Prediction Probability
        ↓
0.5 Threshold
        ↓
Real / AI-Generated
        ↓
Confidence Score
        ↓
Prediction History
```

## 🛠️ Technologies Used

### Machine Learning

- Python
- TensorFlow
- Keras
- NumPy
- PIL
- Scikit-learn

### Deep Learning

- CNN
- Transfer Learning
- ResNet50
- MobileNetV2
- EfficientNetB0
- ImageNet pretrained models

### Web Development

- Django
- HTML
- CSS
- SQLite
- Django ORM

### Development Environment

- Jupyter Notebook
- Google Colab
- Visual Studio Code
- Git
- GitHub

## 📁 Project Structure

```text
AI-Image-Detection-Deep-Learning/
│
├── aiImageDetector/
│   └── Django project configuration
│
├── detector/
│   ├── ai_models.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
│
├── ml_models/
│   ├── baseline_cnn_best.keras
│   ├── mobilenetv2_finetuned_best.keras
│   ├── mobilenetv2_stage1_best.keras
│   ├── resnet50_finetuned_best.h5
│   ├── resnet50_finetuned_best.keras
│   └── resnet50_stage1_best.keras
│
├── notebook/
│   └── 3190951_AI Image Detection.ipynb
│
├── static/
├── templates/
├── manage.py
├── requirements.txt
└── .gitignore
```

## 🚀 Running the Web Application

### 1. Clone the repository

```bash
git clone https://github.com/19jenil/AI-Image-Detection-Deep-Learning.git
cd AI-Image-Detection-Deep-Learning
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply database migrations

```bash
python manage.py migrate
```

### 5. Start the Django server

```bash
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## 📓 Research Notebook

The complete machine learning experimentation workflow is available in:

```text
notebook/3190951_AI Image Detection.ipynb
```

The notebook contains dataset preparation, leakage analysis, preprocessing, augmentation, model training, fine-tuning and evaluation.

## ⚠️ Disclaimer

The reported performance represents the results obtained on the selected CIFAKE dataset and experimental configuration. It should not be interpreted as guaranteed performance on all real-world images or unseen generative AI systems.

## 👨‍💻 Author

**Jenil Patel**

