# 🩺 Skin Condition Detector

A machine learning-powered application for detecting various skin conditions from images using Convolutional Neural Networks (CNN).

## 🧪 Tech Stack

| Component         | Technology/Library              |
|------------------|---------------------------------|
| Programming      | Python 3.x                      |
| AI/ML            | TensorFlow/Keras (CNN)          |
| Image Processing | OpenCV/Pillow                   |
| Frontend/UI      | Streamlit                       |
| Dataset          | Kaggle Skin Disease Datasets    |
| Visualization    | Matplotlib/Seaborn              |

## ✨ Features (MVP)

1. **Photo Upload** - Upload images of skin areas for analysis
2. **Skin Condition Detection** - Classifies into 5 categories:
   - Acne
   - Dryness
   - Healthy
   - Spots
   - Eczema
3. **Care Tips/Recommendations** - Personalized suggestions for each condition
4. **History Tracking** - Stores previous analyses for progress monitoring
5. **Branding** - Professional UI with logo and branding

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd skin_condition_detector
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to the provided local URL (typically http://localhost:8501).

## 📁 Project Structure

```
skin_condition_detector/
├── app.py              # Main Streamlit application
├── model.py            # CNN model implementation
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── models/             # Trained model files
├── data/               # History and data files
└── assets/             # Images and other assets
```

## 🧠 Model Training

The model was trained on Kaggle skin disease datasets with the following preprocessing steps:
1. Image resizing to 224x224 pixels
2. Normalization of pixel values
3. Data augmentation techniques
4. 5-class classification using CNN architecture

## 🔮 Future Enhancements

- Integration with more specialized skin condition datasets
- Real-time camera analysis
- Mobile application version
- Advanced visualization of skin analysis
- Email/SMS notifications for tracking

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any skin concerns.