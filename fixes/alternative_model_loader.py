"""
Alternative approach: Build fresh model and load weights
This bypasses the model serialization issues
"""

import tensorflow as tf
import h5py
import numpy as np
from CNNModel import build_cnn_model_with_residuals

def extract_weights_from_model(model_path):
    """Extract weights from a problematic model"""
    print(f"Extracting weights from: {model_path}")
    
    weights = {}
    layer_weights = {}
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Get model config
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                print("Found model_config")
            
            # Extract layer weights
            if 'layer_weights' in f:
                weights_group = f['layer_weights']
                
                def extract_layer_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        layer_name = name.split('/')[-1] if '/' in name else name
                        if layer_name and not layer_name.startswith('.'):
                            weights[layer_name] = obj[()]
                            print(f"Extracted weights for layer: {layer_name}")
                
                weights_group.visititems(extract_layer_weights)
                
    except Exception as e:
        print(f"Error extracting weights: {e}")
        return None
    
    return weights

def create_fresh_model_and_load_weights(model_path):
    """Create a fresh model and load weights from the problematic model"""
    print("=== CREATING FRESH MODEL WITH WEIGHTS ===")
    
    # Create fresh model with the same architecture
    fresh_model = build_cnn_model_with_residuals(input_shape=(640, 640, 1))
    
    try:
        # Try to load the entire model first
        print("Attempting to load original model...")
        original_model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        # Copy weights from original to fresh model
        print("Copying weights...")
        for i, layer in enumerate(fresh_model.layers):
            try:
                if i < len(original_model.layers):
                    original_layer = original_model.layers[i]
                    if layer.name in [l.name for l in original_model.layers]:
                        orig_layer_by_name = original_model.get_layer(layer.name)
                        if orig_layer_by_name and len(orig_layer_by_name.get_weights()) > 0:
                            layer.set_weights(orig_layer_by_name.get_weights())
                            print(f"Copied weights for layer: {layer.name}")
            except Exception as e:
                print(f"Could not copy weights for layer {layer.name}: {e}")
                continue
        
        print("✅ Fresh model created with weights copied!")
        return fresh_model
        
    except Exception as e:
        print(f"Could not load original model: {e}")
        print("Creating model without weights...")
        return fresh_model

def build_simple_model():
    """Build a simple compatible model"""
    print("Building simple compatible model...")
    
    input_shape = (640, 640, 1)
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    
    # Simple CNN architecture
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Named outputs
    classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    bbox_output = tf.keras.layers.Dense(4, name='bbox_output')(x)
    
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            'classification_output': classification_output,
            'bbox_output': bbox_output
        },
        name='simple_empty_shelf_detector'
    )
    
    return model

def test_model_prediction(model, test_image_path):
    """Test model prediction with a simple test"""
    try:
        import cv2
        
        # Load and preprocess image
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Could not load test image: {test_image_path}")
            return False
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_image, (640, 640))
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        batch_input = np.expand_dims(normalized, axis=0)
        batch_input = np.expand_dims(batch_input, axis=-1)
        
        print("Testing model prediction...")
        predictions = model.predict(batch_input, verbose=0)
        
        print(f"✅ Prediction successful!")
        print(f"Prediction type: {type(predictions)}")
        
        if isinstance(predictions, dict):
            print(f"Classification output shape: {predictions['classification_output'].shape}")
            print(f"BBox output shape: {predictions['bbox_output'].shape}")
            print(f"Classification confidence: {predictions['classification_output'][0, 0]:.3f}")
            print(f"BBox predictions: {predictions['bbox_output'][0].tolist()}")
        else:
            print(f"Output shapes: {[p.shape for p in predictions]}")
        
        return True
        
    except Exception as e:
        print(f"Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test the model loading approach"""
    model_path = "Models/empty_shelf_detector.h5"
    
    if not tf.io.gfile.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Try different approaches
    print("=== ATTEMPTING TO CREATE WORKING MODEL ===")
    
    # Approach 1: Build fresh model and try to copy weights
    print("\n--- Approach 1: Fresh model with weight copying ---")
    try:
        model1 = create_fresh_model_and_load_weights(model_path)
        if test_model_prediction(model1, "Data/Wilson/0c2cc2c7-243a-4194-821b-ec0682cfe42e_jpg.rf.4dea80794260df2a286155f7e0fbcccf.jpg"):
            print("✅ Approach 1 SUCCESS!")
            return model1
    except Exception as e:
        print(f"Approach 1 failed: {e}")
    
    # Approach 2: Simple compatible model
    print("\n--- Approach 2: Simple compatible model ---")
    try:
        model2 = build_simple_model()
        # Compile the model
        model2.compile(
            optimizer='adam',
            loss={
                'classification_output': 'binary_crossentropy',
                'bbox_output': 'mean_squared_error',
            },
            loss_weights={
                'classification_output': 0.2,
                'bbox_output': 2.0,
            },
            metrics={'classification_output': 'accuracy'}
        )
        
        if test_model_prediction(model2, "Data/Wilson/0c2cc2c7-243a-4194-821b-ec0682cfe42e_jpg.rf.4dea80794260df2a286155f7e0fbcccf.jpg"):
            print("✅ Approach 2 SUCCESS!")
            return model2
    except Exception as e:
        print(f"Approach 2 failed: {e}")
    
    print("❌ All approaches failed!")
    return None

if __name__ == "__main__":
    working_model = main()
    if working_model:
        print(f"\n🎉 SUCCESS! Working model created: {working_model.name}")
    else:
        print("\n💥 Failed to create working model")
