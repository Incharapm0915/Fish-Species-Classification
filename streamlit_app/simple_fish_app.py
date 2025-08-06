"""
Simplified Fish Classification Streamlit App
Quick and easy version for immediate testing
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(page_title="🐟 Fish Classifier", page_icon="🐟", layout="wide")

# Title
st.title("🐟 Fish Species Classification")
st.write("Upload a fish image and get instant AI classification!")

@st.cache_resource
def load_model_and_info():
    """Load the best available model"""
    try:
        # Look for models in results directory
        base_dir = os.path.abspath('..')
        models_dir = os.path.join(base_dir, 'results', 'models')
        results_file = os.path.join(base_dir, 'results', 'combined_training_results.json')
        
        # Load results info
        class_names = []
        best_model_name = None
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                class_names = results.get('dataset_info', {}).get('class_names', [])
                best_model_name = results.get('best_model', {}).get('name', None)
        
        # Find and load best model
        if best_model_name and os.path.exists(models_dir):
            model_file = f"{best_model_name}_best.h5"
            model_path = os.path.join(models_dir, model_file)
            
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                return model, class_names, best_model_name
        
        # If best model not found, try to load any available model
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                model = tf.keras.models.load_model(model_path)
                model_name = model_files[0].replace('_best.h5', '').replace('.h5', '')
                return model, class_names, model_name
        
        return None, class_names, None
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, [], None

def preprocess_image(image):
    """Simple image preprocessing"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load model
model, class_names, model_name = load_model_and_info()

if model is None:
    st.error("❌ No trained models found!")
    st.info("Please run the training pipeline first to create models.")
    st.stop()

st.success(f"✅ Loaded model: {model_name}")
st.info(f"🏷️ Can classify {len(class_names)} fish species")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a fish image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a fish"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Show image info
        st.write(f"Image size: {image.size[0]} × {image.size[1]} pixels")

with col2:
    st.header("🤖 Prediction")
    
    if uploaded_file is not None:
        if st.button("🔍 Classify Fish", type="primary"):
            
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(processed_image, verbose=0)[0]
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(predictions)[::-1][:3]
                    
                    # Display results
                    st.success("🎉 Classification Complete!")
                    
                    for i, idx in enumerate(top_indices):
                        confidence = predictions[idx]
                        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
                        
                        # Clean up class name for display
                        display_name = class_name.replace('_', ' ').title()
                        
                        if i == 0:
                            # Main prediction
                            st.markdown(f"""
                            ### 🐟 **{display_name}**
                            **Confidence: {confidence:.1%}**
                            """)
                        else:
                            # Alternative predictions
                            st.write(f"{i+1}. {display_name}: {confidence:.1%}")
                    
                    # Simple visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    top_3_names = [class_names[idx].replace('_', ' ').title() 
                                 for idx in top_indices]
                    top_3_conf = [predictions[idx] for idx in top_indices]
                    
                    bars = ax.barh(top_3_names, top_3_conf, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    ax.set_xlabel('Confidence')
                    ax.set_title('Top 3 Predictions')
                    
                    # Add percentage labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{width:.1%}', ha='left', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download results
                    results_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model': model_name,
                        'predictions': {
                            class_names[idx]: float(predictions[idx]) 
                            for idx in top_indices
                        }
                    }
                    
                    st.download_button(
                        "📄 Download Results",
                        data=json.dumps(results_data, indent=2),
                        file_name=f"fish_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")