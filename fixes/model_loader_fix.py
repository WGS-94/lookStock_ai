"""
Model Loader Fix for Keras compatibility issues
Fixes batch_shape vs batch_input_shape issues and other compatibility problems
"""

import tensorflow as tf
import keras
import h5py
import tempfile
import os
from pathlib import Path

def fix_model_config(model_path):
    """Fix model configuration by converting batch_shape to batch_input_shape"""
    
    def convert_batch_shape_to_input_shape(config_str):
        """Convert batch_shape to batch_input_shape in model config"""
        import json
        
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        
        config = json.loads(config_str)
        
        def convert_recursively(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    if key == 'batch_shape' and isinstance(value, list):
                        new_dict['batch_input_shape'] = value
                    elif key == 'config' and isinstance(value, dict):
                        # Handle layer configs
                        new_dict[key] = convert_recursively(value)
                    else:
                        new_dict[key] = convert_recursively(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_recursively(item) for item in obj]
            return obj
        
        fixed_config = convert_recursively(config)
        return json.dumps(fixed_config).encode('utf-8')
    
    # Create temporary fixed model
    temp_model_path = model_path + '.fixed'
    
    try:
        with h5py.File(model_path, 'r') as src, h5py.File(temp_model_path, 'w') as dst:
            
            def copy_and_fix(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Copy the dataset
                    data = obj[()]
                    
                    # If this is the model_config, fix it
                    if name == 'model_config':
                        print(f"Fixing model_config...")
                        data = convert_batch_shape_to_input_shape(data)
                    
                    # If this is layer config, check and fix
                    elif '/layer_config/' in name:
                        try:
                            config_data = obj[()]
                            if b'batch_shape' in config_data:
                                print(f"Fixing layer config: {name}")
                                config_data = convert_batch_shape_to_input_shape(config_data)
                                data = config_data
                        except:
                            pass
                    
                    dst.create_dataset(name, data=data)
                    
                elif isinstance(obj, h5py.Group):
                    # Create the group and copy attributes
                    grp = dst.create_group(name)
                    for attr_name, attr_value in obj.attrs.items():
                        grp.attrs[attr_name] = attr_value
                        
                    # Copy all children
                    for child_name, child_obj in obj.items():
                        copy_and_fix(f"{name}/{child_name}", child_obj)
            
            copy_and_fix('', src)
        
        return temp_model_path
        
    except Exception as e:
        print(f"Error fixing model: {e}")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        return None

def load_model_with_compatibility(model_path, fallback_path=None):
    """
    Load model with multiple compatibility fixes
    """
    print(f"Attempting to load model from: {model_path}")
    
    # Custom objects for different Keras versions
    custom_objects = {
        'DTypePolicy': tf.keras.mixed_precision.Policy,
    }
    
    # Try different loading strategies
    strategies = [
        # Strategy 1: Normal load with custom objects
        lambda: keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False),
        
        # Strategy 2: Load without compilation
        lambda: keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False),
        
        # Strategy 3: Fix model config and reload
        lambda: load_and_fix_model(model_path, custom_objects),
        
        # Strategy 4: Load with InputLayer patch
        lambda: load_with_inputlayer_patch(model_path, custom_objects),
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            print(f"Trying strategy {i}...")
            model = strategy()
            if model is not None:
                print(f"✅ Model loaded successfully with strategy {i}")
                return model
                
        except Exception as e:
            print(f"Strategy {i} failed: {str(e)[:100]}...")
            continue
    
    # If all strategies fail and fallback exists, try fallback
    if fallback_path and os.path.exists(fallback_path):
        print(f"Trying fallback model: {fallback_path}")
        try:
            model = keras.models.load_model(fallback_path, custom_objects=custom_objects, compile=False, safe_mode=False)
            print(f"✅ Fallback model loaded successfully")
            return model
        except Exception as e:
            print(f"Fallback model failed: {e}")
    
    print("❌ All model loading strategies failed")
    return None

def load_and_fix_model(model_path, custom_objects):
    """Load model by fixing the config first"""
    print("Fixing model configuration...")
    fixed_path = fix_model_config(model_path)
    
    if fixed_path is None:
        raise Exception("Failed to fix model config")
    
    try:
        model = keras.models.load_model(fixed_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        return model
    finally:
        if os.path.exists(fixed_path):
            os.remove(fixed_path)

def load_with_inputlayer_patch(model_path, custom_objects):
    """Load model with InputLayer patch"""
    print("Applying InputLayer patch...")
    
    # Store original init
    original_init = tf.keras.layers.InputLayer.__init__
    
    try:
        def patched_init(self, *args, **kwargs):
            if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
                kwargs['batch_input_shape'] = tuple(kwargs.pop('batch_shape'))
                print("Patched batch_shape to batch_input_shape")
            return original_init(self, *args, **kwargs)
        
        # Apply patch
        tf.keras.layers.InputLayer.__init__ = patched_init
        
        # Try to load
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(model_path, compile=False, safe_mode=False)
            return model
            
    finally:
        # Restore original
        tf.keras.layers.InputLayer.__init__ = original_init

def recompile_model(model):
    """Recompile the model with appropriate settings"""
    try:
        model.compile(
            optimizer='adam',
            loss={
                'classification_output': 'binary_crossentropy',
                'bbox_output': 'mean_squared_error',
            },
            loss_weights={
                'classification_output': 0.2,
                'bbox_output': 2.0,
            },
            metrics={
                'classification_output': 'accuracy'
            }
        )
        print("Model recompiled successfully")
        return True
    except Exception as e:
        print(f"Failed to recompile model: {e}")
        return False

def load_and_prepare_model(model_path, fallback_path=None):
    """
    Complete model loading and preparation pipeline
    """
    print("=== MODEL LOADING AND PREPARATION ===")
    
    # Load model with compatibility fixes
    model = load_model_with_compatibility(model_path, fallback_path)
    
    if model is None:
        print("❌ Failed to load model")
        return None
    
    # Ensure model has named outputs
    if not isinstance(model.outputs, dict):
        print("Converting model outputs to named format...")
        try:
            # Create new model with named outputs
            outputs = {}
            for i, output in enumerate(model.outputs):
                if i == 0:
                    outputs['classification_output'] = output
                elif i == 1:
                    outputs['bbox_output'] = output
            
            new_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=outputs,
                name=model.name
            )
            model = new_model
            print("Model outputs converted successfully")
            
        except Exception as e:
            print(f"Warning: Could not convert outputs to named format: {e}")
    
    # Try to recompile if needed
    try:
        recompile_model(model)
    except Exception as e:
        print(f"Warning: Could not recompile model: {e}")
    
    print(f"✅ Model loaded and prepared successfully!")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output names: {list(model.outputs.keys()) if isinstance(model.outputs, dict) else 'list format'}")
    
    return model

if __name__ == "__main__":
    # Test the model loading
    model_paths = [
        "Models/empty_shelf_detector.h5",
        "empty_shelf_web_app/Models/empty_shelf_detector.h5"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\nTesting model: {model_path}")
            model = load_and_prepare_model(model_path)
            if model:
                print("✅ SUCCESS")
                break
            else:
                print("❌ FAILED")
        else:
            print(f"Model not found: {model_path}")
