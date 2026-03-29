import tensorflow as tf
from tensorflow import keras
import os

# Create directory for saving the model
os.makedirs('models', exist_ok=True)

# Define skin conditions
SKIN_CONDITIONS = ['Acne', 'Dryness', 'Healthy', 'Spots', 'Eczema']

def create_and_save_model():
    """
    Create a simple CNN model and save it for use in our application.
    """
    # Create a simple CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(SKIN_CONDITIONS), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save the model
    model.save('models/skin_model.h5')
    print("Model saved successfully!")
    
    # Print model summary
    model.summary()
    
    return model

if __name__ == "__main__":
    model = create_and_save_model()
    print("Demo model created and saved to models/skin_model.h5")