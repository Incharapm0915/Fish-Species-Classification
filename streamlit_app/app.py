"""
Fish Classification Streamlit Web Application
MINIMAL VERSION - Guaranteed to work on Streamlit Cloud
Only essential packages to avoid conflicts
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import json
import os
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Try to import TensorFlow with proper error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow import failed: {e}")
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🐟 AI Fish Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified but beautiful CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_FILE_SIZE = 10

# Target models
TARGET_MODELS = ['CNN_Scratch_best.h5', 'MobileNet_best.h5']

@st.cache_resource
def load_models_and_info():
    """Load models with simplified cloud-friendly approach"""
    models = {}
    results_info = None
    class_names = []
    
    if not TF_AVAILABLE:
        return models, results_info, class_names
    
    try:
        # Simple path detection for cloud deployment
        possible_paths = [
            'results/compressed_models',
            'results/models', 
            'compressed_models',
            'models'
        ]
        
        for models_dir in possible_paths:
            if os.path.exists(models_dir):
                try:
                    available_files = os.listdir(models_dir)
                    
                    # Look for our target models
                    model_files_to_load = []
                    for target_model in TARGET_MODELS:
                        if target_model in available_files:
                            model_files_to_load.append(target_model)
                        # Check compressed version
                        compressed_name = target_model.replace('.h5', '_compressed.h5')
                        if compressed_name in available_files:
                            model_files_to_load.append(compressed_name)
                    
                    if model_files_to_load:
                        st.info(f"📁 Loading from: {models_dir}")
                        
                        for model_file in model_files_to_load:
                            if 'CNN_Scratch' in model_file:
                                model_name = 'CNN_Scratch'
                                display_name = '🧠 Custom CNN'
                            elif 'MobileNet' in model_file:
                                model_name = 'MobileNet'
                                display_name = '📱 MobileNet'
                            else:
                                continue
                            
                            model_path = os.path.join(models_dir, model_file)
                            
                            try:
                                file_size = os.path.getsize(model_path) / (1024 * 1024)
                                model = tf.keras.models.load_model(model_path, compile=False)
                                
                                models[model_name] = {
                                    'model': model,
                                    'path': model_path,
                                    'file': model_file,
                                    'size_mb': file_size,
                                    'display_name': display_name
                                }
                                
                                st.success(f"✅ Loaded {display_name} ({file_size:.1f}MB)")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to load {model_name}: {e}")
                        
                        if models:
                            break
                            
                except Exception as e:
                    continue
        
        # Load class names
        results_paths = ['results/combined_training_results.json', 'combined_training_results.json']
        for results_file in results_paths:
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results_info = json.load(f)
                    class_names = results_info.get('dataset_info', {}).get('class_names', [])
                    break
                except:
                    continue
        
        # Fallback class names
        if not class_names:
            train_paths = ['data/train', 'train']
            for train_dir in train_paths:
                if os.path.exists(train_dir):
                    try:
                        class_names = [d for d in os.listdir(train_dir) 
                                     if os.path.isdir(os.path.join(train_dir, d))]
                        class_names.sort()
                        break
                    except:
                        continue
        
        return models, results_info, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None, []

def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_array, class_names):
    """Make prediction"""
    try:
        predictions = model.predict(image_array, verbose=0)
        class_probabilities = predictions[0]
        top_indices = np.argsort(class_probabilities)[::-1]
        
        results = []
        max_results = min(5, len(class_names) if class_names else len(class_probabilities))
        
        for i in range(max_results):
            if i < len(top_indices):
                idx = top_indices[i]
                if idx < len(class_names) and class_names:
                    results.append({
                        'class': class_names[idx],
                        'confidence': float(class_probabilities[idx]),
                        'percentage': f"{class_probabilities[idx] * 100:.2f}%"
                    })
                else:
                    results.append({
                        'class': f'Class_{idx}',
                        'confidence': float(class_probabilities[idx]),
                        'percentage': f"{class_probabilities[idx] * 100:.2f}%"
                    })
        return results
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_simple_chart(predictions):
    """Create simple matplotlib chart"""
    if not predictions:
        return None
    
    try:
        classes = [pred['class'][:15] + '...' if len(pred['class']) > 15 else pred['class'] 
                  for pred in predictions[:5]]
        confidences = [pred['confidence'] for pred in predictions[:5]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#11998e']
        bars = ax.barh(classes, confidences, color=colors[:len(classes)])
        
        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{conf:.1%}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Confidence Score', fontweight='bold')
        ax.set_title('🐟 Fish Species Classification Results', fontweight='bold', fontsize=16)
        ax.set_xlim(0, max(confidences) * 1.2 if confidences else 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Apply image enhancements"""
    try:
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        return image
    except Exception as e:
        st.error(f"Error enhancing image: {e}")
        return image

def display_confidence_bars(predictions):
    """Display confidence bars"""
    st.markdown("### 🎯 Prediction Confidence")
    
    for i, pred in enumerate(predictions[:5]):
        confidence = pred['confidence']
        class_name = pred['class'].replace('_', ' ').title()
        
        progress_html = f"""
        <div style="margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #333;">{class_name}</span>
                <span style="font-weight: 700; color: #667eea;">{pred['percentage']}</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

def main():
    """Main application"""
    
    st.markdown('<h1 class="main-header">🐟 AI Fish Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Custom CNN vs Transfer Learning Model Comparison**")
    
    if not TF_AVAILABLE:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ TensorFlow Not Available</h2>
            <p>TensorFlow could not be loaded in this environment.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load models
    with st.spinner("🧠 Loading AI models..."):
        models, results_info, class_names = load_models_and_info()
    
    if not models:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ Models Not Found</h2>
            <p>Looking for CNN_Scratch_best.h5 and MobileNet_best.h5</p>
            <p>Please ensure models are in results/models/ directory</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Debug Info:**")
        st.write(f"Current directory: {os.getcwd()}")
        try:
            st.write(f"Directory contents: {os.listdir('.')}")
            if os.path.exists('results'):
                st.write(f"Results contents: {os.listdir('results')}")
        except:
            st.write("Cannot list directories")
        return
    
    # Success message
    st.markdown(f"""
    <div class="success-card">
        <h3>✅ AI Models Ready!</h3>
        <p>Successfully loaded {len(models)} model{'s' if len(models) > 1 else ''}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model selection
        st.markdown("### 🤖 AI Model Selection")
        model_options = {model_data['display_name']: model_name 
                        for model_name, model_data in models.items()}
        
        selected_display_name = st.selectbox("Choose model:", list(model_options.keys()))
        selected_model = model_options[selected_display_name]
        
        # Model info
        if selected_model in models:
            model_data = models[selected_model]
            st.markdown(f"""
            <div class="info-card">
                <p><strong>📄 File:</strong> {model_data['file']}</p>
                <p><strong>📊 Size:</strong> {model_data['size_mb']:.1f} MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Image enhancement
        st.markdown("### 🎨 Image Enhancement")
        brightness = st.slider("☀️ Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("⚡ Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("🔍 Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # App info
        st.markdown("### 📊 App Statistics")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>🏷️ Fish Species:</strong> {len(class_names) if class_names else 'Unknown'}</p>
            <p><strong>🖼️ Max File Size:</strong> {MAX_FILE_SIZE}MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("## 📤 Upload Your Fish Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help=f"Max file size: {MAX_FILE_SIZE}MB"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ File too large! Max: {MAX_FILE_SIZE}MB (Current: {file_size:.1f}MB)")
                return
            
            try:
                image = Image.open(uploaded_file)
                enhanced_image = enhance_image(image, brightness, contrast, sharpness)
                
                tab1, tab2 = st.tabs(["✨ Enhanced", "📷 Original"])
                
                with tab1:
                    st.image(enhanced_image, caption="Enhanced for AI Analysis", use_container_width=True)
                
                with tab2:
                    st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
                
                st.markdown(f"""
                <div class="info-card">
                    <p><strong>📊 Image Details:</strong></p>
                    <p>• Size: {image.size[0]} × {image.size[1]} pixels</p>
                    <p>• File size: {file_size:.2f} MB</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                return
    
    with col2:
        st.markdown("## 🤖 AI Analysis Results")
        
        if uploaded_file is not None and models and class_names:
            
            if st.button("🔍 Analyze Fish Species", type="primary", use_container_width=True):
                
                with st.spinner("🧠 AI is analyzing your image..."):
                    try:
                        processed_image = preprocess_image(enhanced_image)
                        
                        if processed_image is not None:
                            model_data = models[selected_model]
                            model = model_data['model']
                            
                            predictions = predict_image(model, processed_image, class_names)
                            
                            if predictions:
                                top_prediction = predictions[0]
                                
                                # Main result
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h2>🐟 {top_prediction['class'].replace('_', ' ').title()}</h2>
                                    <h3>Confidence: {top_prediction['percentage']}</h3>
                                    <p>Model: {model_data['display_name']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence bars
                                display_confidence_bars(predictions)
                                
                                # Chart
                                st.markdown("### 📈 Visual Analysis")
                                chart = create_simple_chart(predictions)
                                if chart:
                                    st.pyplot(chart, use_container_width=True)
                                
                                # Export
                                st.markdown("### 💾 Export Results")
                                export_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'image_name': uploaded_file.name,
                                    'model_used': selected_model,
                                    'top_prediction': top_prediction,
                                    'all_predictions': predictions[:5]
                                }
                                
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label="📄 Download Results",
                                    data=json_str,
                                    file_name=f"fish_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ Error processing image")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis error: {e}")
        else:
            st.markdown("""
            <div class="info-card">
                <h3>👆 Ready to Analyze!</h3>
                <p>Upload a fish image to get AI-powered species identification.</p>
                <br>
                <p><strong>🎯 Tips:</strong></p>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Ensure fish is the main subject</li>
                    <li>Try different enhancement settings</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px;'>
        <h3>🐟 AI Fish Species Classifier</h3>
        <p>Built with Streamlit & TensorFlow</p>
        <p>Made with ❤️ for Marine Conservation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
