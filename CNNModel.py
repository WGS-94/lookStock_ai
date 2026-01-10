from tensorflow.keras import layers, Input
import tensorflow as tf

def build_cnn_model_with_residuals(input_shape=(640, 640, 1)):
    """Builds CNN model with residual connections for empty shelf detection"""
    
    input_tensor = Input(shape=input_shape, name='image_input')
    
    # First convolutional block
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same', name='conv1')(input_tensor)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    # Residual block 1
    shortcut = x
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    shortcut = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', name='shortcut1')(shortcut)
    x = layers.Add(name='add1')([x, shortcut])
    
    # Residual block 2
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    shortcut = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', name='shortcut2')(shortcut)
    x = layers.Add(name='add2')([x, shortcut])
    
    # Residual block 3
    shortcut = x
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    shortcut = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', name='shortcut3')(shortcut)
    x = layers.Add(name='add3')([x, shortcut])
    
    # Residual block 4
    shortcut = x
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool5')(x)
    shortcut = layers.Conv2D(512, (1, 1), strides=(2, 2), padding='same', name='shortcut4')(shortcut)
    x = layers.Add(name='add4')([x, shortcut])
    
    # Global pooling and fully connected layers
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Dense layers
    x = layers.Dense(1024, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(512, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.Dense(256, activation='relu', name='dense3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # Output layers with explicit names
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    # Create model with named outputs for proper tensor handling
    model = tf.keras.Model(
        inputs=input_tensor, 
        outputs={
            'classification_output': classification_output,
            'bbox_output': bbox_output
        },
        name='empty_shelf_detector'
    )
    
    return model
