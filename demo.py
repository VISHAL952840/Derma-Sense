"""
Demo script to demonstrate the skin condition detector functionality.
This script shows how to use the different components of the application.
"""

import os
from PIL import Image
import numpy as np
from model import load_model, predict_skin_condition
from utils import get_care_tips, save_history, load_history

def create_sample_image():
    """
    Create a sample image for demonstration purposes.
    """
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

def demo_skin_detection():
    """
    Demonstrate the skin condition detection functionality.
    """
    print("=== Skin Condition Detector Demo ===")
    
    # Load the model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    # Create a sample image
    print("Creating sample image...")
    sample_image = create_sample_image()
    
    # Predict skin condition
    print("Analyzing skin condition...")
    condition, confidence = predict_skin_condition(model, sample_image)
    
    print(f"Detected Condition: {condition}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Get care tips
    print("\n=== Care Tips ===")
    tips = get_care_tips(condition)
    print(tips)
    
    # Save to history
    print("\nSaving to history...")
    save_history("demo_image.jpg", condition, confidence)
    
    # Load history
    print("\n=== History ===")
    history = load_history()
    if history:
        latest_entry = history[-1]
        print(f"Latest entry: {latest_entry['Timestamp']} - {latest_entry['Condition']} ({latest_entry['Confidence']:.2f}%)")
    else:
        print("No history entries found.")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demo_skin_detection()