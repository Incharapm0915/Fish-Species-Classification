"""
Fish Classification Streamlit Web Application
Fixed version with minimal dependencies to avoid conflicts
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import tensorflow as tf
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🐟 Fish Classification App",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .prediction-result {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        color: white;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_FILE_SIZE = 10  # MB

@st.cache_resource
def load_models_and_info():
    """Load trained models and metadata with error handling"""
    models = {}
    results_info = None
    class_names = []
    
    try:
        # Try to load results from the combined pipeline
        base_dir = os.path.abspath('..')
        results_dir = os.path.join(base_dir, 'results')
        models_dir = os.path.join(results_dir, 'models')
        
        # Load results JSON if available
        results_file = os.path.join(results_dir, 'combined_training_results.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results_info = json.load(f)
                class_names = results_info.get('dataset_info', {}).get('class_names', [])
            except Exception as e:
                st.warning(f"Could not load results file: {e}")
        
        # Load available models
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            
            for model_file in model_files:
                model_name = model_file.replace('_best.h5', '').replace('.h5', '')
                model_path = os.path.join(models_dir, model_file)
                
                try:
                    model = tf.keras.models.load_model(model_path)
                    models[model_name] = {
                        'model': model,
                        'path': model_path,
                        'file': model_file
                    }
                    st.success(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    st.warning(f"Could not load {model_name}: {str(e)}")
        
        # If no class names from results, try to infer from training data
        if not class_names:
            train_dir = os.path.join(base_dir, 'data', 'train')
            if os.path.exists(train_dir):
                class_names = [d for d in os.listdir(train_dir) 
                             if os.path.isdir(os.path.join(train_dir, d))]
                class_names.sort()
        
        return models, results_info, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None, []

def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_array, class_names):
    """Make prediction on preprocessed image"""
    try:
        # Get prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get class probabilities
        class_probabilities = predictions[0]
        
        # Get top predictions
        top_indices = np.argsort(class_probabilities)[::-1]
        
        results = []
        for i in range(min(5, len(class_names))):  # Top 5 predictions
            if i < len(top_indices):
                idx = top_indices[i]
                if idx < len(class_names):
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
    """Create simple matplotlib chart instead of plotly"""
    if not predictions:
        return None
    
    # Prepare data
    classes = [pred['class'][:15] + '...' if len(pred['class']) > 15 else pred['class'] 
              for pred in predictions[:5]]  # Truncate long names
    confidences = [pred['confidence'] for pred in predictions[:5]]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, confidences, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
    
    # Add percentage labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{conf:.1%}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Confidence Score')
    ax.set_title('Fish Species Classification Results', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    
    return fig

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

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🐟 Fish Species Classification</h1>', unsafe_allow_html=True)
    st.markdown("**Upload a fish image and get instant species classification using deep learning models!**")
    
    # Load models and info
    with st.spinner("Loading AI models..."):
        models, results_info, class_names = load_models_and_info()
    
    if not models:
        st.error("❌ No trained models found! Please run the training pipeline first.")
        st.info("📋 Instructions:")
        st.code("""
1. Run the combined preprocessing + training pipeline
2. Ensure models are saved in ../results/models/ directory
3. Restart this Streamlit app
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")
        
        # Model selection
        st.subheader("Model Selection")
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Choose AI Model:",
            model_names,
            help="Select which trained model to use for prediction"
        )
        
        # Display model info
        if results_info and selected_model in results_info.get('training_results', {}):
            model_info = results_info['training_results'][selected_model]
            if model_info.get('status') == 'SUCCESS':
                st.success(f"✅ Model Accuracy: {model_info.get('best_val_accuracy', 0):.1%}")
                st.info(f"🎯 Top-3 Accuracy: {model_info.get('best_val_top3_accuracy', 0):.1%}")
        
        st.divider()
        
        # Image enhancement controls
        st.subheader("🎨 Image Enhancement")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        st.divider()
        
        # App info
        st.subheader("ℹ️ App Information")
        st.info(f"🔢 Models Available: {len(models)}")
        st.info(f"🏷️ Fish Species: {len(class_names) if class_names else 'Unknown'}")
        
        if results_info:
            dataset_info = results_info.get('dataset_info', {})
            training_samples = dataset_info.get('total_training_samples', 'Unknown')
            st.info(f"🖼️ Training Images: {training_samples:,}" if isinstance(training_samples, int) else f"🖼️ Training Images: {training_samples}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Fish Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help=f"Maximum file size: {MAX_FILE_SIZE}MB"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ File too large! Maximum size: {MAX_FILE_SIZE}MB (Current: {file_size:.1f}MB)")
                return
            
            # Load and display image
            try:
                image = Image.open(uploaded_file)
                
                # Apply enhancements
                enhanced_image = enhance_image(image, brightness, contrast, sharpness)
                
                # Display images
                tab1, tab2 = st.tabs(["Enhanced Image", "Original Image"])
                
                with tab1:
                    st.image(enhanced_image, caption="Enhanced Image for Prediction", use_column_width=True)
                
                with tab2:
                    st.image(image, caption=f"Original: {uploaded_file.name}", use_column_width=True)
                
                # Image info
                st.info(f"📊 Image Info: {image.size[0]}×{image.size[1]} pixels, {image.mode} mode")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                return
    
    with col2:
        st.subheader("🤖 AI Prediction Results")
        
        if uploaded_file is not None and models and class_names:
            
            # Predict button
            if st.button("🔍 Classify Fish Species", type="primary"):
                
                with st.spinner("🧠 AI is analyzing the image..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(enhanced_image)
                        
                        if processed_image is not None:
                            # Get selected model
                            model_data = models[selected_model]
                            model = model_data['model']
                            
                            # Make prediction
                            predictions = predict_image(model, processed_image, class_names)
                            
                            if predictions:
                                # Display main prediction
                                top_prediction = predictions[0]
                                
                                st.markdown(f"""
                                <div class="prediction-result">
                                    <h2>🐟 {top_prediction['class'].replace('_', ' ').title()}</h2>
                                    <h3>Confidence: {top_prediction['percentage']}</h3>
                                    <p>Model: {selected_model}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display confidence chart using matplotlib
                                st.subheader("📊 Detailed Results")
                                confidence_chart = create_simple_chart(predictions)
                                if confidence_chart:
                                    st.pyplot(confidence_chart)
                                
                                # Alternative predictions
                                if len(predictions) > 1:
                                    st.subheader("🔄 Alternative Classifications")
                                    for i, pred in enumerate(predictions[1:4], 2):  # Show next 3
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.write(f"**{i}. {pred['class'].replace('_', ' ').title()}**")
                                        with col_b:
                                            st.write(f"`{pred['percentage']}`")
                                
                                # Export results
                                st.subheader("💾 Export Results")
                                
                                # Prepare export data
                                export_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'image_name': uploaded_file.name,
                                    'model_used': selected_model,
                                    'top_prediction': top_prediction,
                                    'all_predictions': predictions[:5]
                                }
                                
                                # JSON download
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label="📄 Download Results (JSON)",
                                    data=json_str,
                                    file_name=f"fish_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                        else:
                            st.error("❌ Error processing image for prediction")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
        
        else:
            # Instructions
            st.info("👆 Upload an image to get started!")
            
            if not class_names:
                st.warning("⚠️ Class names not available. Check training results.")
    
    # Model Performance Section (simplified)
    if results_info and results_info.get('training_results'):
        st.divider()
        st.header("📈 Model Performance Overview")
        
        # Get successful models
        training_results = results_info['training_results']
        successful_models = {k: v for k, v in training_results.items() 
                           if v.get('status') == 'SUCCESS'}
        
        if successful_models:
            # Create simple performance table
            st.subheader("🏆 Model Performance Comparison")
            
            performance_data = []
            for model_name, results in successful_models.items():
                performance_data.append({
                    'Model': model_name,
                    'Validation Accuracy': f"{results.get('best_val_accuracy', 0):.3f}",
                    'Top-3 Accuracy': f"{results.get('best_val_top3_accuracy', 0):.3f}",
                    'Training Time (min)': f"{results.get('training_time_minutes', 0):.1f}",
                    'Parameters': f"{results.get('total_params', 0):,}",
                })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Validation Accuracy', ascending=False)
            st.dataframe(performance_df, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🐟 Fish Classification App | Built with Streamlit & TensorFlow</p>
        <p>Upload fish images and get instant AI-powered species identification!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()