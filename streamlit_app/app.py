"""
Cloud-Ready Fish Classification Streamlit App
Optimized for Streamlit Cloud deployment with better error handling
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
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "results/models/MobileNet_best.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Try to import TensorFlow with cloud-specific handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    # Suppress TensorFlow warnings in cloud environment
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🐟 AI Fish Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (same as your beautiful version)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: slideInUp 0.6s ease-out;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
        animation: pulse 2s infinite;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
        transition: transform 0.3s ease;
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3); }
        50% { box-shadow: 0 15px 40px rgba(17, 153, 142, 0.5); }
        100% { box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
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
    
    .confidence-item {
        margin: 0.5rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_FILE_SIZE = 10  # MB

@st.cache_resource
def load_models_and_info():
    """Load trained models with cloud-optimized paths"""
    models = {}
    results_info = None
    class_names = []
    
    if not TF_AVAILABLE:
        return models, results_info, class_names
    
    try:
        # Cloud-optimized path detection
        possible_paths = [
            # For cloud deployment
            os.path.join(os.getcwd(), 'results', 'models'),
            os.path.join(os.getcwd(), 'streamlit_app', '..', 'results', 'models'),
            # For local development  
            os.path.join(os.path.abspath('..'), 'results', 'models'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'models'),
            # Alternative locations
            os.path.join('results', 'models'),
            os.path.join('..', 'results', 'models')
        ]
        
        models_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.listdir(path):
                models_dir = path
                break
        
        if not models_dir:
            st.warning("⚠️ No models directory found. Please ensure models are uploaded to your repository.")
            return models, results_info, class_names
        
        # Load model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        
        if not model_files:
            st.warning("⚠️ No .h5 model files found in the models directory.")
            return models, results_info, class_names
        
        # Load each model with error handling
        for model_file in model_files:
            model_name = model_file.replace('_best.h5', '').replace('.h5', '')
            model_path = os.path.join(models_dir, model_file)
            
            try:
                # Check file size (cloud deployment has limits)
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                if file_size > 100:  # Most cloud services have ~100MB limit
                    st.warning(f"⚠️ Model {model_name} is large ({file_size:.1f}MB). May cause deployment issues.")
                
                model = tf.keras.models.load_model(model_path)
                models[model_name] = {
                    'model': model,
                    'path': model_path,
                    'file': model_file,
                    'size_mb': file_size
                }
                
            except Exception as e:
                st.error(f"❌ Failed to load {model_name}: {str(e)}")
        
        # Try to load results info
        results_paths = [
            os.path.join(os.path.dirname(models_dir), 'combined_training_results.json'),
            os.path.join('results', 'combined_training_results.json'),
            os.path.join('..', 'results', 'combined_training_results.json')
        ]
        
        for results_path in results_paths:
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results_info = json.load(f)
                    class_names = results_info.get('dataset_info', {}).get('class_names', [])
                    break
                except Exception as e:
                    st.warning(f"Could not load results file: {e}")
        
        # Fallback: try to infer class names from data directory
        if not class_names:
            data_paths = [
                os.path.join('data', 'train'),
                os.path.join('..', 'data', 'train'),
                os.path.join(os.getcwd(), 'data', 'train')
            ]
            
            for data_path in data_paths:
                if os.path.exists(data_path):
                    class_names = [d for d in os.listdir(data_path) 
                                 if os.path.isdir(os.path.join(data_path, d))]
                    class_names.sort()
                    break
        
        return models, results_info, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("💡 Make sure your repository includes the 'results/models/' directory with .h5 files")
        return {}, None, []

def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess uploaded image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_array, class_names):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(image_array, verbose=0)
        class_probabilities = predictions[0]
        top_indices = np.argsort(class_probabilities)[::-1]
        
        results = []
        num_results = min(5, len(class_names) if class_names else len(class_probabilities))
        
        for i in range(num_results):
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

def create_beautiful_chart(predictions):
    """Create beautiful matplotlib chart"""
    if not predictions:
        return None
    
    try:
        plt.style.use('default')
        
        classes = [pred['class'][:20] + '...' if len(pred['class']) > 20 else pred['class'] 
                  for pred in predictions[:5]]
        confidences = [pred['confidence'] for pred in predictions[:5]]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#11998e']
        bars = ax.barh(classes, confidences, color=colors[:len(classes)], height=0.6)
        
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{conf:.1%}', ha='left', va='center', fontweight='bold',
                   fontsize=12, color='#333')
        
        ax.set_xlabel('Confidence Score', fontsize=14, fontweight='bold', color='#333')
        ax.set_title('🐟 Fish Species Classification Results', fontsize=18, fontweight='bold', 
                    color='#333', pad=20)
        ax.set_xlim(0, max(confidences) * 1.2 if confidences else 1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ddd')
        ax.spines['bottom'].set_color('#ddd')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(colors='#333', labelsize=11)
        
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
        st.error(f"Error enhancing image: {str(e)}")
        return image

def display_confidence_bars(predictions):
    """Display beautiful confidence bars"""
    st.markdown("### 🎯 Prediction Confidence")
    
    for i, pred in enumerate(predictions[:5]):
        confidence = pred['confidence']
        class_name = pred['class'].replace('_', ' ').title()
        
        progress_html = f"""
        <div class="confidence-item" style="animation-delay: {i * 0.1}s;">
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
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🐟 AI Fish Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Deep Learning for Marine Species Identification</p>', 
                unsafe_allow_html=True)
    
    # Check TensorFlow availability
    if not TF_AVAILABLE:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ TensorFlow Not Available</h2>
            <p>The app is deployed but TensorFlow couldn't be loaded.</p>
            <p>This might be a temporary cloud issue. Try refreshing the page.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load models
    with st.spinner("🧠 Loading AI models..."):
        models, results_info, class_names = load_models_and_info()
    
    if not models:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ No Models Found</h2>
            <p>No trained models were found in the repository.</p>
            <p>Please ensure your repository includes:</p>
            <ul>
                <li>results/models/ directory</li>
                <li>.h5 model files</li>
                <li>Files are properly committed to Git</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Debug information
        st.markdown("### 🔍 Debug Information")
        st.write("Current working directory:", os.getcwd())
        st.write("Files in current directory:", os.listdir('.'))
        
        if os.path.exists('results'):
            st.write("Files in results directory:", os.listdir('results'))
            if os.path.exists('results/models'):
                st.write("Files in models directory:", os.listdir('results/models'))
        
        return
    
    # Success message
    st.markdown(f"""
    <div class="success-card">
        <h3>✅ AI Models Ready!</h3>
        <p>Successfully loaded {len(models)} trained models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rest of your beautiful app code (sidebar, main content, etc.)
    # ... (keeping the same structure as your original app)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        st.markdown("### 🤖 AI Model Selection")
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Choose your AI model:",
            model_names,
            help="Each model has different strengths and accuracies"
        )
        
        # Display model info
        if selected_model in models:
            model_info = models[selected_model]
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Model Info</h4>
                <p><strong>File:</strong> {model_info['file']}</p>
                <p><strong>Size:</strong> {model_info.get('size_mb', 0):.1f}MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎨 Image Enhancement")
        brightness = st.slider("☀️ Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("⚡ Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("🔍 Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        st.markdown("### 📊 App Statistics")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>🔢 Available Models:</strong> {len(models)}</p>
            <p><strong>🏷️ Fish Species:</strong> {len(class_names) if class_names else 'Unknown'}</p>
            <p><strong>🖼️ Max File Size:</strong> {MAX_FILE_SIZE}MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("## 📤 Upload Your Fish Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help=f"Supported formats: PNG, JPG, JPEG, BMP (Max: {MAX_FILE_SIZE}MB)"
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ File too large! Maximum: {MAX_FILE_SIZE}MB (Current: {file_size:.1f}MB)")
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
                    <p>• Mode: {image.mode}</p>
                    <p>• File size: {file_size:.2f} MB</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                return
    
    with col2:
        st.markdown("## 🤖 AI Analysis Results")
        
        if uploaded_file is not None and models:
            
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
                                
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h2>🐟 {top_prediction['class'].replace('_', ' ').title()}</h2>
                                    <h3>Confidence: {top_prediction['percentage']}</h3>
                                    <p>Model: {selected_model}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                display_confidence_bars(predictions)
                                
                                st.markdown("### 📈 Visual Analysis")
                                confidence_chart = create_beautiful_chart(predictions)
                                if confidence_chart:
                                    st.pyplot(confidence_chart, use_container_width=True)
                                
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
                                    label="📄 Download Analysis Report",
                                    data=json_str,
                                    file_name=f"fish_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ Error processing image for prediction")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
        else:
            st.markdown("""
            <div class="info-card">
                <h3>👆 Ready to Analyze!</h3>
                <p>Upload a fish image using the panel on the left to get started with AI-powered species identification.</p>
                <br>
                <p><strong>💡 Tips for best results:</strong></p>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Ensure the fish is the main subject</li>
                    <li>Try different enhancement settings</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px; margin-top: 3rem;'>
        <h3>🐟 AI Fish Species Classifier</h3>
        <p>Powered by Deep Learning • Built with Streamlit & TensorFlow</p>
        <p>Upload fish images and get instant AI-powered species identification!</p>
        <br>
        <p>Made with ❤️ for Marine Conservation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

