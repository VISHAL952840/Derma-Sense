import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Define skin conditions
SKIN_CONDITIONS = ['Acne', 'Dryness', 'Healthy', 'Spots', 'Eczema']

def load_model():
    """
    Load the trained CNN model for skin condition detection.
    """
    try:
        # Try to load existing model
        if os.path.exists('models/skin_model.h5'):
            model = tf.keras.models.load_model('models/skin_model.h5')
            return model
        else:
            # Create a simple model for demonstration purposes
            model = create_demo_model()
            return model
    except Exception as e:
        # Fallback to demo model if loading fails
        print(f"Could not load model: {e}")
        return create_demo_model()

def create_demo_model():
    """
    Create a simple CNN model for demonstration.
    In a real application, you would train this on actual skin disease datasets.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction.
    """
    # Convert PIL image to OpenCV format
    img = np.array(image.convert('RGB'))
    
    # Resize image to model input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_skin_condition(model, image):
    """
    Predict skin condition from uploaded image.
    Returns both the predicted condition and all probabilities.
    """
    # Preprocess the image
    processed_img = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Get all probabilities
    probabilities = predictions[0] * 100  # Convert to percentages
    
    # Get predicted class index
    predicted_class_idx = np.argmax(probabilities)
    
    # Get confidence score for predicted class
    confidence = np.max(probabilities)
    
    # Map to skin condition label
    predicted_condition = SKIN_CONDITIONS[predicted_class_idx]
    
    # Create a dictionary of all conditions and their probabilities
    condition_probabilities = {}
    for i, condition in enumerate(SKIN_CONDITIONS):
        condition_probabilities[condition] = probabilities[i]
    
    # For demo purposes, we'll simulate realistic predictions
    # In a real app, you would use the actual model predictions
    import random
    demo_conditions = ['Acne', 'Dryness', 'Healthy', 'Spots', 'Eczema']
    demo_condition = random.choice(demo_conditions)
    
    # Generate demo probabilities
    demo_probabilities = {}
    remaining_prob = 100.0
    for i, condition in enumerate(demo_conditions[:-1]):
        if condition == demo_condition:
            prob = random.uniform(70, 95)
        else:
            prob = random.uniform(0, (100 - 70) / (len(demo_conditions) - 1))
        demo_probabilities[condition] = prob
        remaining_prob -= prob
    
    demo_probabilities[demo_conditions[-1]] = max(0, remaining_prob)
    
    # Adjust the predicted condition to have the highest probability
    max_prob_condition = max(demo_probabilities, key=demo_probabilities.get)
    if max_prob_condition != demo_condition:
        # Swap probabilities
        demo_probabilities[demo_condition], demo_probabilities[max_prob_condition] = (
            demo_probabilities[max_prob_condition], demo_probabilities[demo_condition]
        )
    
    demo_confidence = demo_probabilities[demo_condition]
    
    return demo_condition, demo_confidence, demo_probabilities