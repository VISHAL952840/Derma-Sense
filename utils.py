import json
import os
from datetime import datetime
import pandas as pd

# History file path
HISTORY_FILE = 'data/history.json'

def get_care_tips(condition):
    """
    Provide care tips based on the detected skin condition.
    """
    tips = {
        "Acne": "1. Cleanse your face twice daily with a gentle cleanser.\n"
                "2. Avoid picking or squeezing pimples.\n"
                "3. Use non-comedogenic products.\n"
                "4. Consider salicylic acid or benzoyl peroxide treatments.\n"
                "5. Consult a dermatologist for persistent acne.",
        
        "Dryness": "1. Moisturize immediately after bathing.\n"
                   "2. Use a humidifier in dry environments.\n"
                   "3. Avoid hot showers and harsh soaps.\n"
                   "4. Apply oils or ceramide-rich creams.\n"
                   "5. Stay hydrated by drinking plenty of water.",
        
        "Healthy": "1. Maintain your current skincare routine.\n"
                   "2. Continue protecting your skin from sun damage.\n"
                   "3. Eat a balanced diet rich in antioxidants.\n"
                   "4. Stay hydrated and get adequate sleep.\n"
                   "5. Regularly check your skin for any changes.",
        
        "Spots": "1. Use broad-spectrum sunscreen daily.\n"
                 "2. Avoid picking at dark spots.\n"
                 "3. Consider vitamin C serums or retinoids.\n"
                 "4. Exfoliate gently 1-2 times per week.\n"
                 "5. Consult a professional for persistent spots.",
        
        "Eczema": "1. Moisturize frequently with fragrance-free products.\n"
                  "2. Identify and avoid triggers.\n"
                  "3. Take short, lukewarm baths.\n"
                  "4. Use mild, hypoallergenic soaps.\n"
                  "5. Consult a dermatologist for severe cases."
    }
    
    return tips.get(condition, "Maintain good skincare habits and consult a dermatologist if you have concerns.")

def save_history(filename, condition, confidence):
    """
    Save the analysis result to history.
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    
    # Create history file if it doesn't exist
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
    
    # Load existing history
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    # Add new entry
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Condition": condition,
        "Confidence": confidence
    }
    
    history.append(new_entry)
    
    # Save updated history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_history():
    """
    Load history data.
    """
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return []
    else:
        return []