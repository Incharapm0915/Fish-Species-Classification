"""
Fish Classification Streamlit Web Application
Enhanced with beautiful UI, animations, and improved user experience
Modified for CNN_Scratch_best.h5 and MobileNet_best.h5 only
STREAMLIT CLOUD OPTIMIZED VERSION
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import warnings
import time
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# Try to import TensorFlow with proper error handling for cloud deployment
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings for cleaner cloud logs
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

# Enhanced CSS for beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styles */
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
    
    /* Card Styles */
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
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3);
    }
    
    /* Animation Keyframes */
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
    
    /* Button Styles */
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress Bar */
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
    
    /* Image Container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-top: 3rem;
    }
    
    /* Confidence Bar Styles */
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

# Specific models to load
TARGET_MODELS = ['CNN_Scratch_best.h5', 'MobileNet_best.h5']

@st.cache_resource
def load_models_and_info():
    """Load only CNN_Scratch and MobileNet models with cloud-optimized error handling"""
    models = {}
    results_info = None
    class_names = []
    
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not available. Please check the deployment environment.")
        return models, results_info, class_names
    
    try:
        # Cloud-optimized path detection
        possible_base_dirs = [
            os.getcwd(),                    # Current directory (Streamlit Cloud)
            os.path.abspath('.'),           # Absolute current directory
            os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(),
        ]
        
        for base_dir in possible_base_dirs:
            # Try compressed models first, then original models
            possible_model_dirs = [
                os.path.join(base_dir, 'results', 'compressed_models'),  # Compressed models (preferred)
                os.path.join(base_dir, 'results', 'models'),             # Original models (backup)
                os.path.join(base_dir, 'models'),                        # Alternative location
                os.path.join(base_dir, 'compressed_models'),             # Alternative location
            ]
            
            for models_dir in possible_model_dirs:
                if os.path.exists(models_dir):
                    try:
                        available_files = os.listdir(models_dir)
                        
                        # Check for exact matches or compressed versions
                        model_files_to_load = []
                        for target_model in TARGET_MODELS:
                            # Check for exact match
                            if target_model in available_files:
                                model_files_to_load.append(target_model)
                            # Check for compressed version
                            compressed_name = target_model.replace('.h5', '_compressed.h5')
                            if compressed_name in available_files:
                                model_files_to_load.append(compressed_name)
                        
                        if model_files_to_load:
                            st.info(f"📁 Loading models from: {models_dir}")
                            
                            for model_file in model_files_to_load:
                                # Clean model name for display
                                if 'CNN_Scratch' in model_file:
                                    model_name = 'CNN_Scratch'
                                    display_name = '🧠 Custom CNN (From Scratch)'
                                elif 'MobileNet' in model_file:
                                    model_name = 'MobileNet'
                                    display_name = '📱 MobileNet (Transfer Learning)'
                                else:
                                    continue
                                
                                model_path = os.path.join(models_dir, model_file)
                                
                                try:
                                    # Check file size for cloud deployment limits
                                    file_size = os.path.getsize(model_path) / (1024 * 1024)
                                    
                                    if file_size > 100:  # Most cloud services limit to ~100MB
                                        st.warning(f"⚠️ Model {model_name} is large ({file_size:.1f}MB). May cause deployment issues.")
                                    
                                    # Load model with memory optimization
                                    with st.spinner(f"Loading {model_name}..."):
                                        model = tf.keras.models.load_model(model_path, compile=False)
                                    
                                    models[model_name] = {
                                        'model': model,
                                        'path': model_path,
                                        'file': model_file,
                                        'size_mb': file_size,
                                        'display_name': display_name
                                    }
                                    
                                    # Show success message with size info
                                    compressed_indicator = "🗜️ " if "_compressed" in model_file else ""
                                    st.success(f"✅ {compressed_indicator}Loaded {display_name} ({file_size:.1f}MB)")
                                    
                                except Exception as e:
                                    st.error(f"❌ Failed to load {model_name}: {str(e)}")
                            
                            # If we found our target models, break out of the loops
                            if len(models) >= 1:  # At least one model loaded
                                break
                    except Exception as e:
                        st.warning(f"Could not access directory {models_dir}: {e}")
                        continue
            
            # If we found models, break out of base_dir loop
            if models:
                break
        
        # Load results JSON if available
        for base_dir in possible_base_dirs:
            results_paths = [
                os.path.join(base_dir, 'results', 'combined_training_results.json'),
                os.path.join(base_dir, 'combined_training_results.json'),
            ]
            
            for results_file in results_paths:
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            results_info = json.load(f)
                        class_names = results_info.get('dataset_info', {}).get('class_names', [])
                        break
                    except Exception as e:
                        st.warning(f"Could not load results file: {e}")
            
            if results_info:
                break
        
        # If no class names from results, try to infer from training data
        if not class_names:
            for base_dir in possible_base_dirs:
                train_paths = [
                    os.path.join(base_dir, 'data', 'train'),
                    os.path.join(base_dir, 'train'),
                ]
                
                for train_dir in train_paths:
                    if os.path.exists(train_dir):
                        try:
                            class_names = [d for d in os.listdir(train_dir) 
                                         if os.path.isdir(os.path.join(train_dir, d))]
                            class_names.sort()
                            break
                        except:
                            continue
                
                if class_names:
                    break
        
        # Show debug information if no models found
        if not models:
            st.error("No models found. Debug information:")
            st.write("Current working directory:", os.getcwd())
            try:
                st.write("Files in current directory:", os.listdir('.'))
                if os.path.exists('results'):
                    st.write("Files in results:", os.listdir('results'))
                    if os.path.exists('results/models'):
                        st.write("Files in results/models:", os.listdir('results/models'))
            except Exception as e:
                st.write(f"Debug error: {e}")
        
        return models, results_info, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None, []

def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess uploaded image for prediction with error handling"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_array, class_names):
    """Make prediction on preprocessed image with enhanced error handling"""
    try:
        # Make prediction with error handling
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

def create_plotly_chart(predictions):
    """Create interactive Plotly chart instead of matplotlib for better cloud performance"""
    if not predictions:
        return None
    
    try:
        # Prepare data
        classes = [pred['class'][:20] + '...' if len(pred['class']) > 20 else pred['class'] 
                  for pred in predictions[:5]]
        confidences = [pred['confidence'] for pred in predictions[:5]]
        percentages = [pred['percentage'] for pred in predictions[:5]]
        
        # Create interactive bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=classes,
                x=confidences,
                orientation='h',
                marker=dict(
                    color=confidences,
                    colorscale='Viridis',
                    colorbar=dict(title="Confidence"),
                ),
                text=percentages,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Confidence: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': '🐟 Fish Species Classification Results',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#333'}
            },
            xaxis_title="Confidence Score",
            yaxis_title="Fish Species",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins, sans-serif", size=12, color="#333"),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def create_matplotlib_fallback(predictions):
    """Fallback matplotlib chart if Plotly fails"""
    if not predictions:
        return None
    
    try:
        # Set seaborn style for better aesthetics
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        classes = [pred['class'][:20] + '...' if len(pred['class']) > 20 else pred['class'] 
                  for pred in predictions[:5]]
        confidences = [pred['confidence'] for pred in predictions[:5]]
        
        # Create figure with custom styling
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart with seaborn colors
        colors = sns.color_palette("viridis", len(classes))
        bars = ax.barh(classes, confidences, color=colors, height=0.6)
        
        # Add value labels with style
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{conf:.1%}', ha='left', va='center', fontweight='bold',
                   fontsize=12, color='#333')
        
        # Customize chart appearance
        ax.set_xlabel('Confidence Score', fontsize=14, fontweight='bold', color='#333')
        ax.set_title('🐟 Fish Species Classification Results', fontsize=18, fontweight='bold', 
                    color='#333', pad=20)
        ax.set_xlim(0, max(confidences) * 1.2 if confidences else 1)
        
        # Remove spines and customize grid
        sns.despine(top=True, right=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Customize tick labels
        ax.tick_params(colors='#333', labelsize=11)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating fallback chart: {e}")
        return None

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Apply image enhancements with error handling"""
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
    """Display beautiful confidence bars with animation"""
    st.markdown("### 🎯 Prediction Confidence")
    
    for i, pred in enumerate(predictions[:5]):
        confidence = pred['confidence']
        class_name = pred['class'].replace('_', ' ').title()
        
        # Create progress bar HTML
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
    
    # Beautiful Header
    st.markdown('<h1 class="main-header">🐟 AI Fish Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comparing Custom CNN vs Transfer Learning Models</p>', 
                unsafe_allow_html=True)
    
    # Check TensorFlow availability first
    if not TF_AVAILABLE:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ TensorFlow Not Available</h2>
            <p>This app requires TensorFlow to function properly.</p>
            <p>The deployment environment may be misconfigured.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load models with loading animation
    with st.spinner("🧠 Loading AI models..."):
        models, results_info, class_names = load_models_and_info()
    
    if len(models) == 0:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ Models Not Found</h2>
            <p>Looking for CNN_Scratch_best.h5 and MobileNet_best.h5</p>
            <p>Please ensure these models are uploaded to your repository.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Expected Models"):
            st.code("""
Required models:
• CNN_Scratch_best.h5 (Custom CNN from scratch)
• MobileNet_best.h5 (Transfer learning model)

Location: results/models/ or results/compressed_models/
Make sure models are committed to your Git repository.
            """)
        
        with st.expander("🔍 Debug Information"):
            st.write("Current directory:", os.getcwd())
            try:
                st.write("Directory contents:", os.listdir('.'))
            except:
                st.write("Cannot list directory contents")
        return
    
    # Success message for loaded models
    st.markdown(f"""
    <div class="success-card">
        <h3>✅ AI Models Ready!</h3>
        <p>Successfully loaded {len(models)} specialized model{'s' if len(models) > 1 else ''}</p>
        <p>🧠 Custom CNN vs 📱 MobileNet Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model Selection with beautiful styling
        st.markdown("### 🤖 AI Model Selection")
        
        # Create a more descriptive model selection
        model_options = {}
        for model_name, model_data in models.items():
            display_name = model_data.get('display_name', model_name)
            model_options[display_name] = model_name
        
        selected_display_name = st.selectbox(
            "Choose your AI model:",
            list(model_options.keys()),
            help="Compare performance between custom CNN and transfer learning"
        )
        
        selected_model = model_options[selected_display_name]
        
        # Display model performance with cards
        if results_info and selected_model in results_info.get('training_results', {}):
            model_info = results_info['training_results'][selected_model]
            if model_info.get('status') == 'SUCCESS':
                accuracy = model_info.get('best_val_accuracy', 0)
                top3_acc = model_info.get('best_val_top3_accuracy', 0)
                training_time = model_info.get('training_time_minutes', 0)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Model Performance</h4>
                    <p><strong>Accuracy:</strong> {accuracy:.1%}</p>
                    <p><strong>Top-3 Accuracy:</strong> {top3_acc:.1%}</p>
                    <p><strong>Training Time:</strong> {training_time:.1f} min</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show model file info
        if selected_model in models:
            model_data = models[selected_model]
            st.markdown(f"""
            <div class="info-card">
                <p><strong>📄 File:</strong> {model_data['file']}</p>
                <p><strong>📊 Size:</strong> {model_data['size_mb']:.1f} MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Image Enhancement Controls
        st.markdown("### 🎨 Image Enhancement")
        st.markdown("*Adjust these settings to improve recognition accuracy*")
        
        brightness = st.slider("☀️ Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("⚡ Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("🔍 Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # App Statistics
        st.markdown("### 📊 Model Comparison")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>🧠 Custom CNN:</strong> Built from scratch</p>
            <p><strong>📱 MobileNet:</strong> Transfer learning</p>
            <p><strong>🏷️ Fish Species:</strong> {len(class_names) if class_names else 'Unknown'}</p>
            <p><strong>🖼️ Max File Size:</strong> {MAX_FILE_SIZE}MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area with improved layout
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("## 📤 Upload Your Fish Image")
        
        # Beautiful file uploader
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help=f"Supported formats: PNG, JPG, JPEG, BMP (Max: {MAX_FILE_SIZE}MB)"
        )
        
        if uploaded_file is not None:
            # File size check
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ File too large! Maximum: {MAX_FILE_SIZE}MB (Current: {file_size:.1f}MB)")
                return
            
            # Load and display image
            try:
                image = Image.open(uploaded_file)
                enhanced_image = enhance_image(image, brightness, contrast, sharpness)
                
                # Beautiful image display with tabs
                tab1, tab2 = st.tabs(["✨ Enhanced", "📷 Original"])
                
                with tab1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(enhanced_image, caption="Enhanced for AI Analysis", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Image info card
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
        
        if uploaded_file is not None and models and class_names:
            
            # Beautiful prediction button
            if st.button("🔍 Analyze Fish Species", type="primary", use_container_width=True):
                
                # Loading animation
                with st.spinner("🧠 AI is analyzing your image..."):
                    time.sleep(1)  # Add slight delay for better UX
                    
                    try:
                        processed_image = preprocess_image(enhanced_image)
                        
                        if processed_image is not None:
                            model_data = models[selected_model]
                            model = model_data['model']
                            
                            predictions = predict_image(model, processed_image, class_names)
                            
                            if predictions:
                                top_prediction = predictions[0]
                                
                                # Beautiful main result card
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h2>🐟 {top_prediction['class'].replace('_', ' ').title()}</h2>
                                    <h3>Confidence: {top_prediction['percentage']}</h3>
                                    <p>Model: {model_data.get('display_name', selected_model)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Beautiful confidence bars
                                display_confidence_bars(predictions)
                                
                                # Chart visualization - Try Plotly first, fallback to matplotlib
                                st.markdown("### 📈 Visual Analysis")
                                
                                try:
                                    # Try interactive Plotly chart
                                    plotly_chart = create_plotly_chart(predictions)
                                    if plotly_chart:
                                        st.plotly_chart(plotly_chart, use_container_width=True)
                                    else:
                                        raise Exception("Plotly chart creation failed")
                                except Exception as e:
                                    # Fallback to matplotlib
                                    st.info("Using fallback visualization...")
                                    matplotlib_chart = create_matplotlib_fallback(predictions)
                                    if matplotlib_chart:
                                        st.pyplot(matplotlib_chart, use_container_width=True)
                                
                                # Export functionality
                                st.markdown("### 💾 Export Results")
                                export_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'image_name': uploaded_file.name,
                                    'model_used': selected_model,
                                    'model_display_name': model_data.get('display_name', selected_model),
                                    'top_prediction': top_prediction,
                                    'all_predictions': predictions[:5],
                                    'environment': 'streamlit_cloud',
                                    'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'N/A'
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
                        # Show debug information for cloud deployment
                        with st.expander("Debug Information"):
                            st.write("Error details:", str(e))
                            st.write("TensorFlow available:", TF_AVAILABLE)
                            if TF_AVAILABLE:
                                st.write("TensorFlow version:", tf.__version__)
        else:
            # Beautiful instructions card
            st.markdown("""
            <div class="info-card">
                <h3>👆 Ready to Analyze!</h3>
                <p>Upload a fish image using the panel on the left to get started with AI-powered species identification.</p>
                <br>
                <p><strong>💡 Compare Two Approaches:</strong></p>
                <ul>
                    <li><strong>🧠 Custom CNN:</strong> Built specifically for fish classification</li>
                    <li><strong>📱 MobileNet:</strong> Pre-trained model with transfer learning</li>
                </ul>
                <br>
                <p><strong>🎯 Tips for best results:</strong></p>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Ensure the fish is the main subject</li>
                    <li>Try different enhancement settings</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Performance Dashboard - Enhanced with Plotly
    if results_info and results_info.get('training_results'):
        st.markdown("---")
        st.markdown("## 📈 Model Performance Comparison")
        
        training_results = results_info['training_results']
        # Filter for only our target models
        our_models = {k: v for k, v in training_results.items() 
                     if k in ['CNN_Scratch', 'MobileNet'] and v.get('status') == 'SUCCESS'}
        
        if our_models:
            # Performance metrics table
            performance_data = []
            for model_name, results in our_models.items():
                display_name = "🧠 Custom CNN" if model_name == "CNN_Scratch" else "📱 MobileNet"
                performance_data.append({
                    'Model': display_name,
                    'Type': 'From Scratch' if model_name == "CNN_Scratch" else 'Transfer Learning',
                    'Validation Accuracy': f"{results.get('best_val_accuracy', 0):.1%}",
                    'Top-3 Accuracy': f"{results.get('best_val_top3_accuracy', 0):.1%}",
                    'Training Time': f"{results.get('training_time_minutes', 0):.1f} min",
                    'Parameters': f"{results.get('total_params', 0):,}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Validation Accuracy', ascending=False)
            
            # Display table with enhanced styling
            st.dataframe(
                performance_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model", help="AI Model Architecture"),
                    "Type": st.column_config.TextColumn("Type", help="Training Approach"),
                    "Validation Accuracy": st.column_config.TextColumn("Accuracy", help="Validation Accuracy"),
                    "Top-3 Accuracy": st.column_config.TextColumn("Top-3", help="Top-3 Accuracy"),
                    "Training Time": st.column_config.TextColumn("Time", help="Training Duration"),
                    "Parameters": st.column_config.TextColumn("Params", help="Model Parameters")
                }
            )
            
            # Interactive comparison charts with Plotly
            if len(our_models) == 2:
                cnn_results = our_models.get('CNN_Scratch', {})
                mobilenet_results = our_models.get('MobileNet', {})
                
                if cnn_results and mobilenet_results:
                    st.markdown("### 📊 Interactive Performance Comparison")
                    
                    # Create comparison charts
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Accuracy comparison
                        try:
                            accuracy_data = {
                                'Model': ['🧠 Custom CNN', '📱 MobileNet'],
                                'Accuracy': [
                                    cnn_results.get('best_val_accuracy', 0) * 100,
                                    mobilenet_results.get('best_val_accuracy', 0) * 100
                                ]
                            }
                            
                            fig_acc = px.bar(
                                accuracy_data, 
                                x='Model', 
                                y='Accuracy',
                                title="Model Accuracy Comparison",
                                color='Accuracy',
                                color_continuous_scale='Viridis',
                                text='Accuracy'
                            )
                            
                            fig_acc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig_acc.update_layout(
                                showlegend=False,
                                height=400,
                                font=dict(family="Poppins, sans-serif"),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig_acc, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating accuracy chart: {e}")
                    
                    with col_chart2:
                        # Training time comparison
                        try:
                            time_data = {
                                'Model': ['🧠 Custom CNN', '📱 MobileNet'],
                                'Training Time (min)': [
                                    cnn_results.get('training_time_minutes', 0),
                                    mobilenet_results.get('training_time_minutes', 0)
                                ]
                            }
                            
                            fig_time = px.bar(
                                time_data, 
                                x='Model', 
                                y='Training Time (min)',
                                title="Training Time Comparison",
                                color='Training Time (min)',
                                color_continuous_scale='Plasma',
                                text='Training Time (min)'
                            )
                            
                            fig_time.update_traces(texttemplate='%{text:.1f}m', textposition='outside')
                            fig_time.update_layout(
                                showlegend=False,
                                height=400,
                                font=dict(family="Poppins, sans-serif"),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig_time, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating time chart: {e}")
                    
                    # Model comparison insights
                    st.markdown("### 🔍 Model Analysis")
                    
                    cnn_acc = cnn_results.get('best_val_accuracy', 0)
                    mobilenet_acc = mobilenet_results.get('best_val_accuracy', 0)
                    cnn_time = cnn_results.get('training_time_minutes', 0)
                    mobilenet_time = mobilenet_results.get('training_time_minutes', 0)
                    
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        if cnn_acc > mobilenet_acc:
                            winner = "🧠 Custom CNN"
                            diff = ((cnn_acc - mobilenet_acc) / mobilenet_acc) * 100
                        else:
                            winner = "📱 MobileNet"
                            diff = ((mobilenet_acc - cnn_acc) / cnn_acc) * 100
                        
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>🏆 Best Accuracy</h4>
                            <p><strong>{winner}</strong></p>
                            <p>{diff:.1f}% better performance</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_analysis2:
                        if cnn_time < mobilenet_time:
                            faster = "🧠 Custom CNN"
                            time_diff = mobilenet_time - cnn_time
                        else:
                            faster = "📱 MobileNet"
                            time_diff = cnn_time - mobilenet_time
                        
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>⚡ Faster Training</h4>
                            <p><strong>{faster}</strong></p>
                            <p>{time_diff:.1f} min faster</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Environment Information for Cloud Deployment
    with st.expander("🔧 Environment Information"):
        st.write("**Deployment Environment:**")
        st.write(f"- TensorFlow: {tf.__version__ if TF_AVAILABLE else 'Not Available'}")
        st.write(f"- NumPy: {np.__version__}")
        st.write(f"- Pandas: {pd.__version__}")
        st.write(f"- Matplotlib: {plt.matplotlib.__version__}")
        st.write(f"- Seaborn: {sns.__version__}")
        st.write(f"- Streamlit: {st.__version__}")
        st.write(f"- Current Directory: {os.getcwd()}")
        st.write(f"- Python Path: {os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'N/A'}")
    
    # Beautiful Footer
    st.markdown("""
    <div class="footer">
        <h3>🐟 AI Fish Species Classifier</h3>
        <p>Comparing Custom CNN vs Transfer Learning Approaches</p>
        <p>🧠 Custom Architecture • 📱 Pre-trained MobileNet</p>
        <br>
        <p>Built with Streamlit & TensorFlow • Optimized for Cloud Deployment</p>
        <p>Made with ❤️ for Marine Conservation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
