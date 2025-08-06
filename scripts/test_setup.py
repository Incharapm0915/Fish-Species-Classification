"""
Test script to verify project setup
"""
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def test_imports():
    """Test if all required libraries are installed"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import keras
        print(f"✅ Keras: {keras.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
        
        print("\n🎉 All imports successful!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU(s) available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️  No GPU detected. Training will use CPU (slower but functional)")

def test_directories():
    """Test if all required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data', 'models', 'src', 'notebooks', 
        'streamlit_app', 'results', 'docs'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            os.makedirs(dir_name, exist_ok=True)
            print(f"   Created {dir_name}/ directory")

def check_dataset():
    """Check if dataset is properly structured"""
    print("\nChecking dataset structure...")
    
    data_splits = ['train', 'test', 'val']
    dataset_found = False
    
    for split in data_splits:
        split_path = os.path.join('data', split)
        if os.path.exists(split_path):
            species_folders = [f for f in os.listdir(split_path) 
                             if os.path.isdir(os.path.join(split_path, f))]
            print(f"✅ {split}/ folder found with {len(species_folders)} species")
            dataset_found = True
        else:
            print(f"⚠️  {split}/ folder not found")
    
    if not dataset_found:
        print("📥 Please place your fish dataset in the data/ folder")
        print("   Expected structure: data/train/, data/test/, data/val/")

def main():
    """Run all tests"""
    print("🐟 Fish Classification Project Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("❌ Setup incomplete. Please install missing packages.")
        return
    
    # Test GPU
    test_gpu()
    
    # Test directories
    test_directories()
    
    # Check dataset
    check_dataset()
    
    print("\n" + "=" * 50)
    print("🚀 Setup verification complete!")
    print("Next steps:")
    print("1. Place your fish dataset in the data/ folder")
    print("2. Run data exploration to understand your dataset")
    print("3. Start with model training scripts")

if __name__ == "__main__":
    main()