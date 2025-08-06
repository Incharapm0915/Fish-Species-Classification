# 🐟 Fish Species Classification Project

An end-to-end deep learning project for classifying fish species from images using CNN and Transfer Learning techniques. Built with TensorFlow/Keras and deployed as a Streamlit web application.

![Fish Classification Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)

## 🎯 Project Overview

This project implements a complete machine learning pipeline for fish species classification:

- **Data Preprocessing**: Automated image preprocessing with smart augmentation
- **Model Training**: CNN from scratch + 5 pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)
- **Model Evaluation**: Comprehensive performance comparison and metrics
- **Web Deployment**: Interactive Streamlit application for real-time predictions

## 🏆 Key Features

- **Multi-Model Architecture**: Compare performance across different model architectures
- **Smart Data Augmentation**: Automatically adjusts augmentation based on dataset size and class balance
- **Real-time Web App**: Upload images and get instant AI-powered predictions
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance visualizations
- **Model Comparison**: Side-by-side comparison of all trained models
- **Export Results**: Download predictions and analysis reports

## 📋 Business Use Cases

1. **Fisheries Management**: Automated species identification for catch monitoring
2. **Marine Research**: Rapid classification for ecological studies  
3. **Commercial Applications**: Quality control in seafood processing
4. **Educational Tools**: Interactive learning applications for marine biology

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/fish-classification.git
cd fish-classification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup project structure:**
```bash
python setup_environment.py
```

### Dataset Setup

1. **Prepare your fish dataset** with the following structure:
```
data/
├── train/
│   ├── species1/
│   ├── species2/
│   └── ...
├── test/
│   ├── species1/
│   ├── species2/
│   └── ...
└── val/
    ├── species1/
    ├── species2/
    └── ...
```

2. **Place images** in JPG/PNG format in respective species folders

### Training Models

Run the complete training pipeline:

```python
# Execute the combined preprocessing + training pipeline
python combined_training_pipeline.py
```

This will:
- ✅ Preprocess your dataset
- ✅ Train multiple models (CNN + Transfer Learning)
- ✅ Compare performance metrics
- ✅ Save best models automatically

### Launch Web App

```bash
cd streamlit_app
streamlit run app.py
```

Visit `http://localhost:8501` to use the web application!

## 📁 Project Structure

```
fish-classification/
├── 📂 data/                          # Dataset (not included in repo)
│   ├── train/
│   ├── test/
│   └── val/
├── 📂 src/                           # Source code
│   ├── data_preprocessing/
│   ├── model_training/
│   ├── evaluation/
│   └── deployment/
├── 📂 streamlit_app/                 # Web application
│   ├── app.py                        # Full-featured app
│   ├── simple_fish_app.py           # Simplified app
│   └── run_streamlit.py             # Launch script
├── 📂 results/                       # Training results (not in repo)
│   ├── models/                       # Saved models (.h5 files)
│   ├── plots/                        # Visualization plots
│   └── reports/                      # Performance reports
├── 📄 combined_training_pipeline.py  # Main training script
├── 📄 setup_environment.py          # Environment setup
├── 📄 config.py                     # Configuration
├── 📄 utils.py                      # Utility functions
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
```

## 🤖 Model Architectures

The project implements and compares multiple deep learning architectures:

### 1. Custom CNN (From Scratch)
- **Architecture**: 3 Conv2D blocks + Global Average Pooling + Dense layers
- **Parameters**: ~1.2M parameters
- **Use Case**: Baseline comparison and custom feature learning

### 2. Transfer Learning Models
| Model | Parameters | Input Size | Strengths |
|-------|------------|------------|-----------|
| **VGG16** | 14.7M | 224×224 | Simple, reliable |
| **ResNet50** | 23.6M | 224×224 | Deep networks, skip connections |
| **MobileNet** | 3.2M | 224×224 | Lightweight, mobile-friendly |
| **InceptionV3** | 21.8M | 299×299 | Multi-scale feature extraction |
| **EfficientNetB0** | 4.0M | 224×224 | Optimal accuracy/efficiency balance |

## 📊 Performance Metrics

The project evaluates models using:

- **Accuracy**: Overall classification accuracy
- **Top-3 Accuracy**: Correct class in top 3 predictions
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training Time**: Model efficiency comparison

## 🎨 Data Augmentation Strategy

Smart augmentation based on dataset characteristics:

- **Light** (>5K images): Horizontal flip + rotation
- **Moderate** (2K-5K images): + brightness + zoom + shifts
- **Heavy** (<2K images): + shear + advanced transformations

## 🔧 Configuration

Key parameters in `config.py`:

```python
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 📱 Streamlit Web Application

### Features:
- **Image Upload**: Drag & drop or browse for fish images
- **Real-time Prediction**: Instant AI classification
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Model Comparison**: Switch between different trained models
- **Image Enhancement**: Adjust brightness, contrast, sharpness
- **Export Results**: Download predictions as JSON

### Usage:
1. Upload a fish image (PNG/JPG)
2. Select model for prediction
3. Click "Classify Fish Species"
4. View results and confidence scores
5. Download results if needed

## 🛠️ Development

### Adding New Models

1. **Add model creation function** in `src/model_training/`
2. **Update model list** in training pipeline
3. **Test and evaluate** performance
4. **Update documentation**

### Custom Dataset

1. **Organize images** in train/test/val structure
2. **Update class names** in config
3. **Run preprocessing** pipeline
4. **Train models** and evaluate

## 📈 Results Analysis

The training pipeline generates:

- **Performance comparison charts**
- **Training history plots**
- **Confusion matrices**
- **Model comparison table**
- **Detailed JSON reports**

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app/app.py
```

### Cloud Deployment
- **Streamlit Cloud**: Push to GitHub and deploy
- **Heroku**: Use provided Dockerfile
- **AWS/GCP**: Deploy as containerized application

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Create Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras** for deep learning framework
- **Streamlit** for web application framework
- **Pre-trained models** from TensorFlow Hub
- **Fish datasets** from various marine biology sources

## 📞 Contact

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## 🔥 Quick Demo

```python
# Load trained model
model = load_model('results/models/EfficientNetB0_best.h5')

# Classify fish image
prediction = classify_fish('path/to/fish/image.jpg')
print(f"Species: {prediction['species']} (Confidence: {prediction['confidence']:.2%})")
```

**Star ⭐ this repo if you found it helpful!**
