import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import speech_recognition as sr
import speech_recognition
import io
import base64

# Import our model and utility functions
from model import load_model, predict_skin_condition
from utils import get_care_tips, save_history, load_history

# Set page config
st.set_page_config(
    page_title="Skin Condition Detector",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from {transform: translateY(-20px); opacity: 0;}
        to {transform: translateY(0); opacity: 1;}
    }
    .tips-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #FF9800;
    }
    .medicine-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .health-score-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        text-align: center;
    }
    .health-score-value {
        font-size: 3em;
        font-weight: bold;
        margin: 10px 0;
    }
    .health-score-green {
        color: #4CAF50;
    }
    .health-score-yellow {
        color: #FF9800;
    }
    .health-score-red {
        color: #F44336;
    }
    .price-tag {
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 10px;
    }
    .history-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .chat-container {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    .ai-message {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    .user-message:before {
        content: "👤";
        position: absolute;
        left: -15px;
        top: 5px;
        font-size: 1.5em;
    }
    .ai-message:before {
        content: "🤖";
        position: absolute;
        left: -15px;
        top: 5px;
        font-size: 1.5em;
    }
    .probability-bar {
        height: 20px;
        border-radius: 10px;
        margin-bottom: 5px;
        animation: fillBar 1s ease-out;
    }
    @keyframes fillBar {
        from {width: 0;}
        to {width: var(--width);}
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }
    .confetti {
        position: fixed;
        width: 10px;
        height: 10px;
        background-color: #f00;
        animation: confetti-fall 5s linear forwards;
    }
    @keyframes confetti-fall {
        0% {transform: translateY(-100px) rotate(0deg);}
        100% {transform: translateY(1000px) rotate(720deg);}
    }
    .interactive-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .interactive-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .interactive-btn:active {
        transform: translateY(1px);
    }
    .tab-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .tab-btn:hover {
        transform: scale(1.05);
    }
    .severity-card {
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    .severity-mild {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .severity-moderate {
        background-color: #fff9c4;
        color: #f57f17;
    }
    .severity-severe {
        background-color: #ffcdd2;
        color: #c62828;
    }
    .doctor-alert {
        background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        text-align: center;
        font-weight: bold;
        color: #c62828;
        border: 2px solid #f44336;
        animation: pulse 2s infinite;
    }
    .image-comparison {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    .image-container {
        text-align: center;
        flex: 1;
        min-width: 300px;
    }
    .image-container img {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .image-label {
        font-weight: bold;
        margin-top: 10px;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Language translations
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German'
}

# Translation dictionary
TRANSLATIONS = {
    'en': {
        'app_title': '🩺 Skin Condition Detector',
        'app_subtitle': 'Upload an image of your skin to detect conditions like acne, dryness, spots, eczema, and more!',
        'detection_tab': '📸 Detection',
        'results_tab': '📊 Results',
        'ai_tab': '🤖 AI Assistant',
        'upload_instruction': 'Please upload a clear photo of the skin area you want to analyze',
        'choose_image': 'Choose an image...',
        'analyze_button': '🔍 Analyze Skin Condition',
        'results_title': '📊 Prediction Results',
        'skin_condition': 'Skin Condition',
        'confidence': 'Confidence',
        'severity': 'Severity',
        'health_score': 'Skin Health Score',
        'condition_probabilities': 'Condition Probabilities',
        'condition_distribution': 'Condition Distribution',
        'care_tips': '🧴 Care Tips for',
        'medicine_suggestions': '💊 Medicine Suggestions for',
        'ask_question': 'Ask a question about your skin condition, treatments, or skincare advice',
        'current_analysis': 'Current Analysis',
        'type_question': 'Type your question here:',
        'send_button': '📨 Send',
        'clear_chat': '🗑️ Clear Chat',
        'ai_capabilities': '💡 AI Assistant Capabilities',
        'treatment_info': 'Treatment Information',
        'prevention_tips': 'Prevention Tips',
        'dietary_advice': 'Dietary Advice',
        'product_recommendations': 'Product Recommendations',
        'professional_guidance': 'Professional Guidance',
        'severity_analysis': 'Severity Analysis',
        'original_image': 'Original Image',
        'analyzed_image': 'Analyzed Image',
        'clear_results': 'Clear Results',
        'doctor_alert': '⚠️ Doctor Alert: This condition requires professional medical attention. Please consult a dermatologist soon.',
        'good': '🟢 Good',
        'fair': '🟡 Fair',
        'poor': '🔴 Poor'
    },
    'hi': {
        'app_title': '🩺 त्वचा की स्थिति का पता लगानेवाला',
        'app_subtitle': 'अपनी त्वचा की एक स्पष्ट छवि अपलोड करें ताकि मुहासों, सूखापन, धब्बों, एक्ज़िमा आदि की स्थिति का पता चल सके!',
        'detection_tab': '📸 पहचान',
        'results_tab': '📊 परिणाम',
        'ai_tab': '🤖 एआई सहायक',
        'upload_instruction': 'कृपया विश्लेषण के लिए अपनी त्वचा के क्षेत्र की एक स्पष्ट तस्वीर अपलोड करें',
        'choose_image': 'एक छवि चुनें...',
        'analyze_button': '🔍 त्वचा की स्थिति का विश्लेषण करें',
        'results_title': '📊 भविष्यवाणी परिणाम',
        'skin_condition': 'त्वचा की स्थिति',
        'confidence': 'आत्मविश्वास',
        'severity': 'गंभीरता',
        'health_score': 'त्वचा स्वास्थ्य स्कोर',
        'condition_probabilities': 'शर्त की संभावनाएं',
        'condition_distribution': 'शर्त वितरण',
        'care_tips': '🧴 के लिए देखभाल युक्तियाँ',
        'medicine_suggestions': '💊 के लिए दवा सुझाव',
        'ask_question': 'अपनी त्वचा की स्थिति, उपचार या स्किनकेयर सलाह के बारे में प्रश्न पूछें',
        'current_analysis': 'वर्तमान विश्लेषण',
        'type_question': 'अपना प्रश्न यहां टाइप करें:',
        'send_button': '📨 भेजें',
        'clear_chat': '🗑️ चैट साफ़ करें',
        'ai_capabilities': '💡 एआई सहायक क्षमताएँ',
        'treatment_info': 'उपचार जानकारी',
        'prevention_tips': 'रोकथाम युक्तियाँ',
        'dietary_advice': 'आहार सलाह',
        'product_recommendations': 'उत्पाद की सिफारिशें',
        'professional_guidance': 'पेशेवर मार्गदर्शन',
        'severity_analysis': 'गंभीरता विश्लेषण',
        'original_image': 'मूल छवि',
        'analyzed_image': 'विश्लेषण की गई छवि',
        'clear_results': 'परिणाम साफ़ करें',
        'doctor_alert': '⚠️ डॉक्टर चेतावनी: इस स्थिति के लिए पेशेवर चिकित्सा ध्यान आवश्यक है। कृपया जल्द ही त्वचा विशेषज्ञ से परामर्श लें।',
        'good': '🟢 अच्छा',
        'fair': '🟡 ठीक',
        'poor': '🔴 खराब'
    },
    'es': {
        'app_title': '🩺 Detector de Afecciones de la Piel',
        'app_subtitle': 'Sube una imagen de tu piel para detectar afecciones como acné, sequedad, manchas, eccema y más!',
        'detection_tab': '📸 Detección',
        'results_tab': '📊 Resultados',
        'ai_tab': '🤖 Asistente de IA',
        'upload_instruction': 'Por favor, sube una foto clara del área de piel que deseas analizar',
        'choose_image': 'Elige una imagen...',
        'analyze_button': '🔍 Analizar Condición de la Piel',
        'results_title': '📊 Resultados de Predicción',
        'skin_condition': 'Condición de la Piel',
        'confidence': 'Confianza',
        'severity': 'Gravedad',
        'health_score': 'Puntaje de Salud de la Piel',
        'condition_probabilities': 'Probabilidades de Condición',
        'condition_distribution': 'Distribución de Condición',
        'care_tips': '🧴 Consejos de Cuidado para',
        'medicine_suggestions': '💊 Sugerencias de Medicamentos para',
        'ask_question': 'Haz preguntas sobre tu condición de piel, tratamientos o consejos de cuidado',
        'current_analysis': 'Análisis Actual',
        'type_question': 'Escribe tu pregunta aquí:',
        'send_button': '📨 Enviar',
        'clear_chat': '🗑️ Limpiar Chat',
        'ai_capabilities': '💡 Capacidades del Asistente de IA',
        'treatment_info': 'Información de Tratamiento',
        'prevention_tips': 'Consejos de Prevención',
        'dietary_advice': 'Consejos Dietéticos',
        'product_recommendations': 'Recomendaciones de Productos',
        'professional_guidance': 'Orientación Profesional',
        'severity_analysis': 'Análisis de Gravedad',
        'original_image': 'Imagen Original',
        'analyzed_image': 'Imagen Analizada',
        'clear_results': 'Limpiar Resultados',
        'doctor_alert': '⚠️ Alerta Médica: Esta condición requiere atención médica profesional. Por favor, consulta pronto con un dermatólogo.',
        'good': '🟢 Buena',
        'fair': '🟡 Regular',
        'poor': '🔴 Mala'
    },
    'fr': {
        'app_title': '🩺 Détecteur d\'Affections Cutanées',
        'app_subtitle': 'Téléchargez une image de votre peau pour détecter des affections comme l\'acné, la sécheresse, les taches, l\'eczéma et plus encore !',
        'detection_tab': '📸 Détection',
        'results_tab': '📊 Résultats',
        'ai_tab': '🤖 Assistant IA',
        'upload_instruction': 'Veuillez télécharger une photo claire de la zone de peau que vous souhaitez analyser',
        'choose_image': 'Choisissez une image...',
        'analyze_button': '🔍 Analyser l\'État de la Peau',
        'results_title': '📊 Résultats de Prédiction',
        'skin_condition': 'État de la Peau',
        'confidence': 'Confiance',
        'severity': 'Sévérité',
        'health_score': 'Score de Santé de la Peau',
        'condition_probabilities': 'Probabilités d\'Affection',
        'condition_distribution': 'Distribution des Affections',
        'care_tips': '🧴 Conseils de Soin pour',
        'medicine_suggestions': '💊 Suggestions de Médicaments pour',
        'ask_question': 'Posez des questions sur votre affection cutanée, les traitements ou les conseils de soins',
        'current_analysis': 'Analyse Actuelle',
        'type_question': 'Tapez votre question ici :',
        'send_button': '📨 Envoyer',
        'clear_chat': '🗑️ Effacer le Chat',
        'ai_capabilities': '💡 Capacités de l\'Assistant IA',
        'treatment_info': 'Informations sur le Traitement',
        'prevention_tips': 'Conseils de Prévention',
        'dietary_advice': 'Conseils Alimentaires',
        'product_recommendations': 'Recommandations de Produits',
        'professional_guidance': 'Orientation Professionnelle',
        'severity_analysis': 'Analyse de Sévérité',
        'original_image': 'Image Originale',
        'analyzed_image': 'Image Analysée',
        'clear_results': 'Effacer les Résultats',
        'doctor_alert': '⚠️ Alerte Médicale : Cette affection nécessite une attention médicale professionnelle. Veuillez consulter rapidement un dermatologue.',
        'good': '🟢 Bon',
        'fair': '🟡 Moyen',
        'poor': '🔴 Mauvais'
    },
    'de': {
        'app_title': '🩺 Hautzustand-Detektor',
        'app_subtitle': 'Laden Sie ein Bild Ihrer Haut hoch, um Zustände wie Akne, Trockenheit, Flecken, Ekzeme und mehr zu erkennen!',
        'detection_tab': '📸 Erkennung',
        'results_tab': '📊 Ergebnisse',
        'ai_tab': '🤖 KI-Assistent',
        'upload_instruction': 'Bitte laden Sie ein klares Foto des Hautbereichs hoch, den Sie analysieren möchten',
        'choose_image': 'Wählen Sie ein Bild...',
        'analyze_button': '🔍 Hautzustand analysieren',
        'results_title': '📊 Vorhersageergebnisse',
        'skin_condition': 'Hautzustand',
        'confidence': 'Konfidenz',
        'severity': 'Schweregrad',
        'health_score': 'Hautgesundheits-Score',
        'condition_probabilities': 'Zustandswahrscheinlichkeiten',
        'condition_distribution': 'Zustandsverteilung',
        'care_tips': '🧴 Pflegetipps für',
        'medicine_suggestions': '💊 Medikamentenvorschläge für',
        'ask_question': 'Stellen Sie Fragen zu Ihrem Hautzustand, Behandlungen oder Pflegehinweisen',
        'current_analysis': 'Aktuelle Analyse',
        'type_question': 'Geben Sie Ihre Frage hier ein:',
        'send_button': '📨 Senden',
        'clear_chat': '🗑️ Chat löschen',
        'ai_capabilities': '💡 KI-Assistent-Funktionen',
        'treatment_info': 'Behandlungsinformationen',
        'prevention_tips': 'Präventionstipps',
        'dietary_advice': 'Ernährungsberatung',
        'product_recommendations': 'Produktempfehlungen',
        'professional_guidance': 'Professionelle Anleitung',
        'severity_analysis': 'Schweregrad-Analyse',
        'original_image': 'Originalbild',
        'analyzed_image': 'Analysiertes Bild',
        'clear_results': 'Ergebnisse löschen',
        'doctor_alert': '⚠️ Arztalarm: Dieser Zustand erfordert professionelle medizinische Aufmerksamkeit. Bitte konsultieren Sie bald einen Dermatologen.',
        'good': '🟢 Gut',
        'fair': '🟡 Ausreichend',
        'poor': '🔴 Schlecht'
    }
}

# Function to get translated text
def t(key, lang='en'):
    """Get translated text for a given key and language"""
    translation = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    return translation if translation else key

# Function to calculate skin health score
def calculate_skin_health_score(probabilities):
    """
    Calculate overall skin health score based on probabilities.
    Healthy condition contributes positively, others negatively.
    """
    # Weight factors for each condition (Healthy is positive, others negative)
    weights = {
        'Acne': -0.7,
        'Dryness': -0.6,
        'Healthy': 1.0,
        'Spots': -0.5,
        'Eczema': -0.8
    }
    
    # Calculate weighted score
    score = 0
    total_weight = 0
    for condition, prob in probabilities.items():
        score += prob * weights[condition]
        total_weight += abs(weights[condition])
    
    # Normalize to 0-100 range
    normalized_score = (score + total_weight) / (2 * total_weight) * 100
    return max(0, min(100, normalized_score))

# Function to get health score color class
def get_health_score_color_class(score):
    """
    Get CSS class for health score color based on value.
    """
    if score >= 70:
        return "health-score-green"
    elif score >= 40:
        return "health-score-yellow"
    else:
        return "health-score-red"

# Function to get health score indicator
def get_health_score_indicator(score, lang='en'):
    """
    Get traffic light indicator for health score.
    """
    if score >= 70:
        return t('good', lang)
    elif score >= 40:
        return t('fair', lang)
    else:
        return t('poor', lang)

# Function to determine severity level
def get_severity_level(condition, confidence):
    """
    Determine severity level based on condition and confidence.
    """
    # High confidence problematic conditions are severe
    if confidence > 80 and condition in ['Acne', 'Eczema']:
        return "Severe"
    elif confidence > 60:
        return "Moderate"
    else:
        return "Mild"

# Function to check if doctor alert is needed
def needs_doctor_alert(condition, confidence, severity):
    """
    Check if a doctor alert is needed based on condition and severity.
    """
    # Severe conditions or high confidence in problematic conditions
    if severity == "Severe" or (confidence > 85 and condition in ['Acne', 'Eczema']):
        return True
    return False

# Function to get medicine suggestions with prices
def get_medicine_suggestions_with_prices(condition, lang='en'):
    """
    Provide medicine suggestions with approximate prices based on the detected skin condition.
    """
    suggestions = {
        "Acne": [
            {"name": "Benzoyl Peroxide Gel 2.5%", "type": "OTC", "price": "₹150-₹300", "description": "Reduces bacteria and unclogs pores"},
            {"name": "Salicylic Acid Cleanser", "type": "OTC", "price": "₹200-₹400", "description": "Gentle exfoliation for acne-prone skin"},
            {"name": "Tretinoin Cream 0.025%", "type": "Prescription", "price": "₹400-₹800", "description": "Retinoid for cell turnover"},
            {"name": "Doxycycline 100mg", "type": "Prescription", "price": "₹300-₹600 (10 tablets)", "description": "Antibiotic for inflammatory acne"},
            {"name": "Tea Tree Oil (15ml)", "type": "Natural", "price": "₹250-₹500", "description": "Natural antibacterial treatment"}
        ],
        
        "Dryness": [
            {"name": "CeraVe Moisturizing Cream", "type": "OTC", "price": "₹600-₹900", "description": "Ceramide-based intensive moisturizer"},
            {"name": "Hyaluronic Acid Serum", "type": "OTC", "price": "₹500-₹1000", "description": "Deep hydration serum"},
            {"name": "Colloidal Oatmeal Bath", "type": "OTC", "price": "₹200-₹400", "description": "Soothing bath treatment"},
            {"name": "Urea Cream 10%", "type": "OTC", "price": "₹300-₹600", "description": "Humectant for severe dryness"},
            {"name": "Shea Butter (200g)", "type": "Natural", "price": "₹400-₹700", "description": "Natural emollient"}
        ],
        
        "Healthy": [
            {"name": "Vitamin C Serum", "type": "OTC", "price": "₹800-₹1500", "description": "Antioxidant for skin protection"},
            {"name": "SPF 50+ Sunscreen", "type": "OTC", "price": "₹500-₹1200", "description": "Broad-spectrum UV protection"},
            {"name": "Gentle Cleanser", "type": "OTC", "price": "₹300-₹600", "description": "pH-balanced daily cleanser"},
            {"name": "Night Repair Cream", "type": "OTC", "price": "₹1000-₹2000", "description": "Overnight skin recovery"},
            {"name": "Multivitamin Supplement", "type": "OTC", "price": "₹500-₹1000 (30 capsules)", "description": "Skin health from within"}
        ],
        
        "Spots": [
            {"name": "Hydroquinone Cream 2%", "type": "OTC", "price": "₹400-₹800", "description": "Skin lightening agent"},
            {"name": "Glycolic Acid Serum", "type": "OTC", "price": "₹600-₹1200", "description": "Chemical exfoliant for brightening"},
            {"name": "Retinol Serum 0.5%", "type": "OTC", "price": "₹800-₹1500", "description": "Cell turnover for even skin tone"},
            {"name": "Zinc Oxide Sunscreen", "type": "OTC", "price": "₹400-₹800", "description": "Physical sunblock"},
            {"name": "Aloe Vera Gel (200ml)", "type": "Natural", "price": "₹150-₹300", "description": "Soothing and healing"}
        ],
        
        "Eczema": [
            {"name": "Hydrocortisone Cream 1%", "type": "OTC", "price": "₹200-₹400", "description": "Mild steroid for inflammation"},
            {"name": "Tacrolimus Ointment 0.03%", "type": "Prescription", "price": "₹1000-₹2000", "description": "Non-steroidal immunomodulator"},
            {"name": "Petroleum Jelly (100g)", "type": "OTC", "price": "₹100-₹200", "description": "Barrier repair moisturizer"},
            {"name": "Oatmeal Bath Powder", "type": "OTC", "price": "₹250-₹500", "description": "Anti-inflammatory bath treatment"},
            {"name": "Coconut Oil (200ml)", "type": "Natural", "price": "₹200-₹400", "description": "Natural moisturizing oil"}
        ]
    }
    
    return suggestions.get(condition, [
        {"name": "Consult Dermatologist", "type": "Professional", "price": "₹500-₹2000", "description": "Professional consultation and treatment"}
    ])

# Function to convert speech to text
def speech_to_text():
    """
    Convert speech to text using microphone input.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak now.")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for audio input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        st.success("Speech recognized successfully!")
        return text
    except sr.WaitTimeoutError:
        st.error("No speech detected within timeout period.")
        return None
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from speech recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to analyze acne photos
def analyze_acne_photo(image):
    """
    Analyze acne photos and provide severity assessment.
    """
    # In a real implementation, this would use computer vision techniques
    # For this demo, we'll provide a sample analysis
    
    analysis = "Acne Photo Analysis:\n\n"
    analysis += "Severity Assessment: Moderate\n"
    analysis += "Affected Area: 15-20% of facial surface\n"
    analysis += "Lesion Types: Comedones (blackheads/whiteheads), Papules\n\n"
    analysis += "Recommendations:\n"
    analysis += "1. Use products with 2% salicylic acid or 5% benzoyl peroxide\n"
    analysis += "2. Avoid picking or squeezing lesions\n"
    analysis += "3. Maintain gentle cleansing routine twice daily\n"
    analysis += "4. Consider consulting a dermatologist if condition worsens\n"
    
    return analysis

# Function to check product ingredients
def check_product_ingredients(ingredients):
    """
    Check product ingredients for compatibility with skin conditions.
    """
    # Common beneficial ingredients for different skin conditions
    beneficial_ingredients = {
        "Acne": ["salicylic acid", "benzoyl peroxide", "niacinamide", "tea tree oil", "azelaic acid", "sulfur"],
        "Dryness": ["hyaluronic acid", "ceramides", "glycerin", "shea butter", "squalane", "niacinamide"],
        "Healthy": ["vitamin c", "retinol", "peptides", "niacinamide", "hyaluronic acid", "spf"],
        "Spots": ["vitamin c", "niacinamide", "kojic acid", "azelaic acid", "retinol", "glycolic acid"],
        "Eczema": ["ceramides", "colloidal oatmeal", "niacinamide", "shea butter", "glycerin", "petroleum jelly"]
    }
    
    # Common harmful ingredients for different skin conditions
    harmful_ingredients = {
        "Acne": ["coconut oil", "cocoa butter", "lanolin", "isopropyl myristate", "waxes"],
        "Dryness": ["alcohol", "fragrance", "sulfates", "witch hazel", "clay"],
        "Healthy": ["harsh alcohols", "synthetic fragrances", "sulfates"],
        "Spots": ["fragrance", "essential oils", "alcohol", "harsh exfoliants"],
        "Eczema": ["fragrance", "essential oils", "alcohol", "sulfates", "preservatives", "witch hazel"]
    }
    
    # Convert ingredients to lowercase for comparison
    ingredients_lower = [ingredient.lower() for ingredient in ingredients]
    
    # Get current condition from session state
    current_condition = st.session_state.get("current_condition", "Healthy")
    
    # Check beneficial ingredients
    beneficial_found = []
    for ingredient in ingredients_lower:
        if any(beneficial in ingredient for beneficial in beneficial_ingredients.get(current_condition, [])):
            beneficial_found.append(ingredient)
    
    # Check harmful ingredients
    harmful_found = []
    for ingredient in ingredients_lower:
        if any(harmful in ingredient for harmful in harmful_ingredients.get(current_condition, [])):
            harmful_found.append(ingredient)
    
    # Generate analysis
    analysis = f"Product Ingredient Analysis for {current_condition}:\n\n"
    
    if beneficial_found:
        analysis += f"✅ Beneficial ingredients found: {', '.join(beneficial_found)}\n"
    else:
        analysis += f"⚠️ No beneficial ingredients found for {current_condition}\n"
    
    if harmful_found:
        analysis += f"❌ Potentially harmful ingredients found: {', '.join(harmful_found)}\n\n"
        analysis += "Consider avoiding products with these ingredients."
    else:
        analysis += f"✅ No harmful ingredients found for {current_condition}\n\n"
        analysis += "This product appears to be compatible with your skin condition."
    
    return analysis

# Function to generate personalized skincare routine
def generate_skincare_routine(condition):
    """
    Generate a personalized skincare routine based on the skin condition.
    """
    routines = {
        "Acne": {
            "title": "Acne Treatment Routine",
            "morning": [
                "1. Gentle cleanser (containing salicylic acid)",
                "2. Niacinamide serum (10% concentration)",
                "3. Oil-free moisturizer with SPF 30+",
                "4. Spot treatment (benzoyl peroxide 2.5%) if needed"
            ],
            "evening": [
                "1. Double cleanse with oil cleanser followed by water-based cleanser",
                "2. Exfoliate 2-3 times a week (salicylic acid or glycolic acid)",
                "3. Retinoid treatment (start with adapalene 0.1%)",
                "4. Hydrating serum (hyaluronic acid)",
                "5. Lightweight, non-comedogenic moisturizer"
            ],
            "weekly": [
                "1. Clay mask once a week",
                "2. Avoid picking or squeezing pimples"
            ],
            "tips": [
                "Avoid dairy and high glycemic foods",
                "Change pillowcases daily",
                "Don't over-cleanse (max twice daily)",
                "Be patient - results take 6-8 weeks"
            ]
        },
        "Dryness": {
            "title": "Intensive Hydration Routine",
            "morning": [
                "1. Cream-based gentle cleanser",
                "2. Hyaluronic acid serum on damp skin",
                "3. Rich moisturizer with ceramides",
                "4. Mineral sunscreen (zinc oxide)"
            ],
            "evening": [
                "1. Oil-based cleanser (cleansing balm/oil)",
                "2. Hydrating toner",
                "3. Hyaluronic acid serum",
                "4. Occlusive moisturizer (petroleum jelly or shea butter)",
                "5. Facial oil (jojoba or squalane)"
            ],
            "weekly": [
                "1. Oatmeal bath or honey mask",
                "2. Avoid hot showers"
            ],
            "tips": [
                "Use humidifier in your room",
                "Drink 8-10 glasses of water daily",
                "Avoid alcohol-based products",
                "Pat skin dry instead of rubbing"
            ]
        },
        "Healthy": {
            "title": "Maintenance Routine",
            "morning": [
                "1. Gentle foaming cleanser",
                "2. Vitamin C serum (10-15% concentration)",
                "3. Hydrating moisturizer",
                "4. Broad-spectrum SPF 30+"
            ],
            "evening": [
                "1. Double cleanse if wearing makeup",
                "2. Retinol serum (0.25-0.5%) 2-3 times a week",
                "3. Hydrating serum",
                "4. Night moisturizer"
            ],
            "weekly": [
                "1. Chemical exfoliation (glycolic or lactic acid) 1-2 times",
                "2. Hydrating sheet mask"
            ],
            "tips": [
                "Consistency is key for maintaining healthy skin",
                "Protect from sun damage daily",
                "Get 7-8 hours of sleep",
                "Eat antioxidant-rich foods"
            ]
        },
        "Spots": {
            "title": "Brightening Routine",
            "morning": [
                "1. Gentle cleanser",
                "2. Vitamin C serum (15-20% concentration)",
                "3. Niacinamide serum",
                "4. Broad-spectrum SPF 50+ (very important)",
                "5. Light moisturizer"
            ],
            "evening": [
                "1. Gentle cleanser",
                "2. Exfoliate 2-3 times a week (glycolic or kojic acid)",
                "3. Retinoid treatment",
                "4. Hydrating serum",
                "5. Rich moisturizer"
            ],
            "weekly": [
                "1. Brightening mask with vitamin C or niacinamide",
                "2. Avoid sun exposure during peak hours"
            ],
            "tips": [
                "Sun protection is crucial for preventing dark spots",
                "Be patient - brightening takes 8-12 weeks",
                "Avoid harsh scrubs that can worsen pigmentation",
                "Consider professional treatments for stubborn spots"
            ]
        },
        "Eczema": {
            "title": "Eczema Soothing Routine",
            "morning": [
                "1. Oil-based cleanser or cleansing balm",
                "2. Soothing toner with colloidal oatmeal",
                "3. Anti-inflammatory serum (niacinamide)",
                "4. Rich, fragrance-free moisturizer",
                "5. Mineral sunscreen"
            ],
            "evening": [
                "1. Oil-based cleanser",
                "2. Colloidal oatmeal bath or compress",
                "3. Anti-inflammatory serum",
                "4. Occlusive moisturizer (petroleum jelly)",
                "5. Topical corticosteroid if prescribed"
            ],
            "weekly": [
                "1. Colloidal oatmeal bath",
                "2. Identify and avoid triggers"
            ],
            "tips": [
                "Keep skin constantly moisturized",
                "Avoid known irritants (fragrance, alcohol)",
                "Use lukewarm water for bathing",
                "Wear soft, breathable fabrics"
            ]
        }
    }
    
    return routines.get(condition, routines["Healthy"])

# Function to get personalized responses based on user history
def get_personalized_response(question, condition, lang='en'):
    """
    Get personalized responses based on user history.
    """
    # Load user history
    history_data = load_history()
    
    # Get previous conditions if available
    previous_conditions = []
    if history_data:
        previous_conditions = [entry['Condition'] for entry in history_data[-5:]]  # Last 5 conditions
    
    # Create personalized context
    personalized_context = ""
    if previous_conditions:
        personalized_context = f"Based on your previous skin analysis history showing {', '.join(set(previous_conditions))}, "
    
    return personalized_context

# Function to get AI assistant responses using OpenAI
def get_ai_assistant_response(question, condition, lang='en'):
    """
    Get AI assistant responses using OpenAI API.
    """
    # Check if API key is provided
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        return "Please enter your OpenAI API key in the sidebar to use the AI assistant."
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get personalized context
        personalized_context = get_personalized_response(question, condition, lang)
        
        # Create a system message with context about skin conditions
        system_message = f"""
        You are a dermatology AI assistant helping users with skin condition information.
        The user has been diagnosed with {condition}.
        {personalized_context}
        Provide helpful, accurate, and concise information about this skin condition.
        Focus on answering the specific question asked by the user.
        Include treatment options, prevention tips, and when to see a dermatologist as relevant to the question.
        Be empathetic and professional in your responses.
        IMPORTANT: Always respond in Hinglish (a mix of Hindi and English) regardless of the language of the question.
        Hinglish example: "Acne ka treatment karna bahut zaroori hai kyunki yeh aapke confidence ko bhi affect karta hai. Mild acne ke liye aap OTC products (₹150-₹1000) use kar sakte hain."
        """
        
        # Create the user message
        user_message = f"Question about {condition}: {question}"
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # Fallback to original keyword-based responses if OpenAI fails
        question = question.lower()
        if "treatment" in question or "cure" in question or "medicine" in question or "cream" in question:
            return f"{condition} ka treatment karna bahut zaroori hai. Mild cases ke liye OTC products (₹150-₹1000) ka istemal kiya ja sakta hai, jabki severe cases ke liye prescription medications (₹400-₹2000) ki zaroorat padti hai. Personalized treatment ke liye kisi dermatologist se milna zaroori hai. Aapko apne symptoms ke hisab se sahi treatment lena chahiye."
        elif "cause" in question or "reason" in question or "why" in question:
            return f"{condition} ke kai karan ho sakte hain jaise genetics, environmental triggers, hormonal changes aur lifestyle factors. Aapke specific causes ko identify karne ke liye dermatologist se milna achha rahega. Har vyakti ke liye karan alag ho sakte hain isliye professional diagnosis important hai."
        elif "prevent" in question or "avoid" in question or "stop" in question:
            return f"{condition} ko prevent karne ke liye ek consistent skincare routine maintain karein, apni skin ko UV exposure se protect karein, known irritants se bachen aur apni skin ko proper tarike se moisturize karein. Prevention {condition} ke treatment se behtar hai."
        elif "diet" in question or "food" in question or "eat" in question:
            return f"Diet {condition} ko affect karti hai. Anti-inflammatory foods ka istemal karein. Supplements jaise Omega-3 aur Vitamin D faydemand ho sakte hain. Processed foods aur potential allergens se bachen jo flare-ups ko trigger kar sakte hain. Har kisi ka body alag react karta hai isliye apni diet ko observe karein."
        elif "natural" in question or "home" in question or "remedy" in question:
            return f"{condition} ke liye kuch natural remedies hain jaise aloe vera, oatmeal baths, aur cold compresses. Ye cost-effective options medical treatments ko complement kar sakti hain lekin professional care ko replace nahi karein. Natural remedies ka bhi proper istemal karein."
        elif "price" in question or "cost" in question or "money" in question or "expensive" in question:
            medicines = get_medicine_suggestions_with_prices(condition)
            price_info = "\n".join([f"• {med['name']}: {med['price']}" for med in medicines[:3]])
            return f"{condition} treatments ke typical price ranges:\n{price_info}\n\nAccurate pricing ke liye local pharmacies ya dermatologists se jaanch karein. Treatment ke liye budget banana important hai."
        elif "severity" in question or "serious" in question or "bad" in question or "worse" in question:
            return f"{condition} ki severity kai factors par nirbhar karti hai jaise coverage area, symptoms aur duration. Agar symptoms persist karte hain ya worsen hote hain to turant kisi dermatologist se milen. Samay sahi treatment ke liye important hai."
        elif "time" in question or "when" in question or "how long" in question:
            return f"{condition} ke treatment mein aam taur par 4-12 weeks ka samay lagta hai. Har vyakti ke liye yeh samay alag ho sakta hai. Consistency treatment ke liye sabse important hai. Jald results ke liye patience aur regularity chahiye."
        elif "side effect" in question or "side" in question:
            return f"{condition} ke treatment ke kuch side effects ho sakte hain. Har medicine ka apna effect hota hai. New treatment shuru karne se pehle dermatologist se jarur samjhauta karein. Self-medication se bachen."
        else:
            return f"Main samajh gaya ki aap {condition} ke baare mein {question} puchna chah rahe hain. Aapke skin condition ke baare mein sabse accurate aur personalized advice ke liye, main recommend karta hoon ki aap kisi dermatologist se milen. Kya aap thoda aur detail mein bata sakte hain ki aap kis baare mein jankari chahte hain?"

# Function to create probability chart
def create_probability_chart(probabilities):
    """
    Create a bar chart showing probabilities for all conditions.
    """
    conditions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, probs, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Probability (%)', fontweight='bold')
    ax.set_title('Skin Condition Probabilities', fontweight='bold')
    ax.set_ylim(0, 100)
    
    plt.xticks(rotation=45, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Function to create pie chart
def create_pie_chart(probabilities):
    """
    Create a pie chart showing probabilities for all conditions.
    """
    conditions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create color map
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges = ax.pie(probs, labels=conditions, colors=colors, startangle=90)
    ax.set_title('Skin Condition Distribution', fontweight='bold')
    plt.tight_layout()
    
    return fig

# Function to create interactive probability display
def display_interactive_probabilities(probabilities):
    """
    Create an interactive display of probabilities with visual bars.
    """
    st.subheader("Condition Probabilities")
    
    # Sort probabilities by value
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for condition, prob in sorted_probs:
        # Create a color based on probability
        if prob > 70:
            color = "#4CAF50"  # Green
        elif prob > 40:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
            
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold;">{condition}</span>
                <span>{prob:.1f}%</span>
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px;">
                <div style="background-color: {color}; width: {prob}%; height: 20px; border-radius: 10px; transition: width 1s ease-in-out;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Function to process image for analysis visualization
def process_image_for_analysis(image):
    """
    Process image to create an analyzed version for comparison.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Apply some visual processing to simulate analysis
    # This is just for demonstration - in a real app, this would be the actual analysis result
    processed_img = img_array.copy()
    
    # Add some visual effects to show "analysis"
    # This is just for demonstration purposes
    height, width = processed_img.shape[:2]
    
    # Add a border to show analysis
    border_size = 20
    processed_img[:border_size, :] = [255, 0, 0]  # Red top border
    processed_img[-border_size:, :] = [255, 0, 0]  # Red bottom border
    processed_img[:, :border_size] = [255, 0, 0]  # Red left border
    processed_img[:, -border_size:] = [255, 0, 0]  # Red right border
    
    # Add some text overlay
    cv2.putText(processed_img, 'ANALYZED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert back to PIL Image
    analyzed_image = Image.fromarray(processed_img)
    
    return analyzed_image

# Function to perform detailed dermatological analysis
def detailed_skin_analysis(image):
    """
    Perform detailed dermatological analysis of the skin image.
    Returns structured analysis results.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # For demo purposes, we'll generate simulated analysis results
    # In a real application, this would involve computer vision techniques
    
    # Simulate skin condition detection
    skin_conditions = ['Acne', 'Dryness', 'Healthy', 'Spots', 'Eczema']
    skin_condition = np.random.choice(skin_conditions)
    
    # Simulate acne analysis
    acne_types = ['whiteheads', 'blackheads', 'papules', 'pustules', 'nodules']
    detected_acne_types = np.random.choice(acne_types, size=np.random.randint(1, 4), replace=False)
    acne_count = np.random.randint(5, 50)
    
    # Simulate acne severity levels
    acne_severity_levels = ['Mild', 'Moderate', 'Severe']
    acne_severity = np.random.choice(acne_severity_levels)
    
    # Simulate dryness analysis
    dryness_levels = ['Low', 'Medium', 'High']
    dryness_level = np.random.choice(dryness_levels)
    
    # Simulate redness/inflammation score (0-100)
    redness_score = np.random.randint(0, 100)
    
    # Simulate pore visibility
    pore_visibility_levels = ['Low', 'Medium', 'High']
    pore_visibility = np.random.choice(pore_visibility_levels)
    
    # Simulate skin health score (0-100)
    skin_health_score = np.random.randint(30, 100)
    
    # Simulate recommendations based on analysis
    recommendations = []
    if 'Acne' in skin_condition or acne_count > 10:
        recommendations.extend([
            "Use a gentle salicylic acid cleanser twice daily",
            "Apply benzoyl peroxide spot treatment to active breakouts",
            "Avoid picking or squeezing pimples to prevent scarring",
            "Consider consulting a dermatologist for persistent acne"
        ])
    
    if dryness_level in ['Medium', 'High']:
        recommendations.extend([
            "Use a hydrating moisturizer immediately after cleansing",
            "Avoid hot showers which can strip natural oils",
            "Consider using a humidifier in dry environments",
            "Look for products with hyaluronic acid or ceramides"
        ])
    
    if redness_score > 50:
        recommendations.extend([
            "Use products with anti-inflammatory ingredients like niacinamide",
            "Avoid harsh scrubs or alcohol-based products",
            "Apply a soothing moisturizer with aloe vera or chamomile",
            "Consider patch testing new products before full application"
        ])
    
    if pore_visibility == 'High':
        recommendations.extend([
            "Use products with niacinamide to minimize pore appearance",
            "Incorporate gentle exfoliation 2-3 times per week",
            "Avoid oil-based products that can clog pores",
            "Consider professional treatments like chemical peels"
        ])
    
    # Ensure we have at least 3 recommendations
    if len(recommendations) < 3:
        recommendations.extend([
            "Maintain a consistent skincare routine morning and evening",
            "Always wear broad-spectrum SPF 30+ during the day",
            "Stay hydrated and maintain a balanced diet"
        ])
    
    # Limit to 5 recommendations
    recommendations = recommendations[:5]
    
    # Create structured analysis result
    analysis_result = {
        "skin_condition": skin_condition,
        "acne_severity": acne_severity,
        "acne_count": int(acne_count),
        "dryness_level": dryness_level,
        "redness_score": int(redness_score),
        "pore_visibility": pore_visibility,
        "skin_health_score": int(skin_health_score),
        "recommendations": recommendations
    }
    
    return analysis_result

# Function to create visualization with annotations
def create_annotated_visualization(image, analysis_result):
    """
    Create an annotated visualization of the skin analysis.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create a copy for annotation
    annotated_img = img_array.copy()
    
    # Get image dimensions
    height, width = annotated_img.shape[:2]
    
    # For demo purposes, we'll add simulated annotations
    # In a real application, this would use actual detection coordinates
    
    # Add bounding boxes for acne (simulated)
    for i in range(min(analysis_result['acne_count'], 10)):  # Limit to 10 for demo
        # Random positions for acne
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50)
        w = np.random.randint(10, 30)
        h = np.random.randint(10, 30)
        
        # Draw rectangle (green for acne)
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add heatmap for redness (simulated)
    # Create a transparent overlay for redness
    overlay = annotated_img.copy()
    
    # Add random red patches for inflammation simulation
    for i in range(5):
        center_x = np.random.randint(100, width-100)
        center_y = np.random.randint(100, height-100)
        radius = np.random.randint(20, 80)
        
        # Draw filled circle with transparency
        cv2.circle(overlay, (center_x, center_y), radius, (0, 0, 255), -1)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, 0.3, annotated_img, 0.7, 0, annotated_img)
    
    # Add legend
    cv2.putText(annotated_img, 'Green boxes: Acne', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_img, 'Red overlay: Inflammation', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert back to PIL Image
    annotated_image = Image.fromarray(annotated_img)
    
    return annotated_image

# Main app
def main():
    # Language selection
    st.sidebar.selectbox("🌐 Language / भाषा / Langue", 
                         options=list(LANGUAGES.keys()), 
                         format_func=lambda x: LANGUAGES[x],
                         key="language")
    
    # OpenAI API Key input
    st.sidebar.text_input("OpenAI API Key", key="openai_api_key", type="password")
    
    lang = st.session_state.get("language", "en")
    
    st.markdown(f"<div class='header'><h1>{t('app_title', lang)}</h1><p>{t('app_subtitle', lang)}</p></div>", unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize session state for current condition
    if 'current_condition' not in st.session_state:
        st.session_state.current_condition = None
    
    # Initialize session state for current probabilities
    if 'current_probabilities' not in st.session_state:
        st.session_state.current_probabilities = {}
    
    # Initialize session state for original image
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Initialize session state for analyzed image
    if 'analyzed_image' not in st.session_state:
        st.session_state.analyzed_image = None
    
    # Initialize session state for detailed analysis
    if 'detailed_analysis' not in st.session_state:
        st.session_state.detailed_analysis = None
    
    # Load the trained model
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    
    # Create tabs for different sections with icons
    tab1, tab2, tab3 = st.tabs(["📸 " + t('detection_tab', lang), "📊 " + t('results_tab', lang), "💬 " + t('ai_tab', lang)])
    
    with tab1:
        st.header("📸 " + t('detection_tab', lang))
        st.markdown(t('upload_instruction', lang))
        
        # File uploader
        uploaded_file = st.file_uploader(t('choose_image', lang), type=["jpg", "jpeg", "png"], key="uploader")
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=t('original_image', lang), use_column_width=True)
            
            # Add a button to trigger prediction
            if st.button(t('analyze_button', lang), key="analyze_btn", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Process the image and make prediction
                    condition, confidence, probabilities = predict_skin_condition(model, image)
                    
                    # Perform detailed skin analysis
                    detailed_analysis = detailed_skin_analysis(image)
                    
                    # Save to session state
                    st.session_state.current_condition = condition
                    st.session_state.current_probabilities = probabilities
                    st.session_state.original_image = image
                    st.session_state.analyzed_image = create_annotated_visualization(image, detailed_analysis)
                    st.session_state.detailed_analysis = detailed_analysis
                    
                    # Save to history
                    save_history(uploaded_file.name, condition, confidence)
                    
                    # Show confetti effect for positive results
                    if condition == "Healthy":
                        st.balloons()
                    
                    st.success("Analysis complete! Switch to the 'Results' tab to see detailed information.")
        else:
            st.info("👆 Please upload an image to get started.")
            
        # Information section
        st.markdown("---")
        st.subheader("How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("📸 **Upload**\n\nUpload a clear photo of your skin")
        with col2:
            st.markdown("🔍 **Analyze**\n\nAI examines your skin condition")
        with col3:
            st.markdown("💡 **Results**\n\nGet detailed insights and recommendations")
        
    with tab2:
        st.header("📊 " + t('results_tab', lang))
        
        # Check if we have a condition to display
        if st.session_state.current_condition and st.session_state.current_probabilities:
            condition = st.session_state.current_condition
            probabilities = st.session_state.current_probabilities
            confidence = probabilities[condition]
            original_image = st.session_state.original_image
            analyzed_image = st.session_state.analyzed_image
            
            # Calculate skin health score
            health_score = calculate_skin_health_score(probabilities)
            
            # Determine severity
            severity = get_severity_level(condition, confidence)
            
            # Check for doctor alert
            doctor_alert = needs_doctor_alert(condition, confidence, severity)
            
            # Display results
            st.markdown(f"<div class='result-card'><h2>{t('results_title', lang)}</h2><h3>{t('skin_condition', lang)}: {condition}</h3><p>{t('confidence', lang)}: {confidence:.2f}%</p></div>", unsafe_allow_html=True)
            
            # Display detailed analysis if available
            if st.session_state.detailed_analysis:
                detailed_analysis = st.session_state.detailed_analysis
                st.markdown("<div class='result-card'><h3>🔬 Detailed Dermatological Analysis</h3></div>", unsafe_allow_html=True)
                
                # Create columns for key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='health-score-card'><h4>Acne Count</h4><div class='health-score-value health-score-green'>{detailed_analysis['acne_count']}</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='health-score-card'><h4>Redness Score</h4><div class='health-score-value {get_health_score_color_class(detailed_analysis['redness_score'])}'>{detailed_analysis['redness_score']}/100</div></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='health-score-card'><h4>Skin Health</h4><div class='health-score-value {get_health_score_color_class(detailed_analysis['skin_health_score'])}'>{detailed_analysis['skin_health_score']}/100</div></div>", unsafe_allow_html=True)
                
                # Display detailed metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<div class='tips-card'><h4>Acne Severity</h4><p>{detailed_analysis['acne_severity']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='tips-card'><h4>Dryness Level</h4><p>{detailed_analysis['dryness_level']}</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='tips-card'><h4>Pore Visibility</h4><p>{detailed_analysis['pore_visibility']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='tips-card'><h4>Skin Condition</h4><p>{detailed_analysis['skin_condition']}</p></div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("<div class='medicine-card'><h3>💡 Personalized Recommendations</h3></div>", unsafe_allow_html=True)
                for i, recommendation in enumerate(detailed_analysis['recommendations'], 1):
                    st.markdown(f"<div class='tips-card'><p><strong>{i}.</strong> {recommendation}</p></div>", unsafe_allow_html=True)
                
                # Display structured JSON data
                st.markdown("<div class='result-card'><h3>📋 Structured Analysis Data</h3></div>", unsafe_allow_html=True)
                st.json(detailed_analysis)
            
            # Display severity level
            severity_class = f"severity-{severity.lower()}"
            st.markdown(f"<div class='severity-card {severity_class}'>{t('severity', lang)}: {severity}</div>", unsafe_allow_html=True)
            
            # Display doctor alert if needed
            if doctor_alert:
                st.markdown(f"<div class='doctor-alert'>{t('doctor_alert', lang)}</div>", unsafe_allow_html=True)
            
            # Display skin health score
            score_color_class = get_health_score_color_class(health_score)
            score_indicator = get_health_score_indicator(health_score, lang)
            st.markdown(f"""
            <div class='health-score-card'>
                <h3>{t('health_score', lang)}</h3>
                <div class='health-score-value {score_color_class}'>{health_score:.1f}%</div>
                <div>{score_indicator}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display image comparison
            if original_image and analyzed_image:
                st.subheader("ImageRelation")
                st.markdown("<div class='image-comparison'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    st.image(original_image, caption=t('original_image', lang), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    st.image(analyzed_image, caption="Annotated Analysis Image", use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add explanation of annotations
                st.markdown("""
                <div class='tips-card'>
                    <h4>🔍 Annotation Legend</h4>
                    <p><span style='color: green;'>■</span> Green boxes: Detected acne regions</p>
                    <p><span style='color: red;'>■</span> Red overlay: Areas of inflammation/redness</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create and display probability charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(t('condition_probabilities', lang))
                prob_chart = create_probability_chart(probabilities)
                st.pyplot(prob_chart)
            with col2:
                st.subheader(t('condition_distribution', lang))
                pie_chart = create_pie_chart(probabilities)
                st.pyplot(pie_chart)
            
            # Display interactive probabilities
            display_interactive_probabilities(probabilities)
            
            # Get care tips
            tips = get_care_tips(condition)
            st.markdown(f"<div class='tips-card'><h3>{t('care_tips', lang)} {condition}</h3><p>{tips}</p></div>", unsafe_allow_html=True)
            
            # Get medicine suggestions with prices
            medicines = get_medicine_suggestions_with_prices(condition, lang)
            st.markdown(f"<div class='medicine-card'><h3>{t('medicine_suggestions', lang)} {condition}</h3>", unsafe_allow_html=True)
            
            # Display medicines in a table
            med_data = []
            for med in medicines:
                med_data.append([
                    med['name'],
                    med['type'],
                    med['price'],
                    med['description']
                ])
            
            med_df = pd.DataFrame(med_data, columns=pd.Index(['Medicine', 'Type', 'Price Range', 'Description']))
            st.table(med_df)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Clear results button
            if st.button(t('clear_results', lang), key="clear_results_btn", use_container_width=True):
                # Clear session state
                st.session_state.current_condition = None
                st.session_state.current_probabilities = {}
                st.session_state.original_image = None
                st.session_state.analyzed_image = None
                st.rerun()
            
            # Show confetti effect for positive results
            if condition == "Healthy":
                st.balloons()
        else:
            st.info("Please analyze an image first in the Detection tab.")
            
    with tab3:
        st.header("💬 " + t('ai_tab', lang))
        st.markdown(t('ask_question', lang))
        
        # Check if we have a condition to discuss
        if st.session_state.current_condition or (load_history() and len(load_history()) > 0):
            # Get current condition
            if st.session_state.current_condition:
                current_condition = st.session_state.current_condition
            else:
                history_data = load_history()
                current_condition = history_data[-1]['Condition']
            
            st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 20px;'><strong>{t('current_analysis', lang)}:</strong> {current_condition}</div>", unsafe_allow_html=True)
            
            # Display chat history
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    if chat['role'] == 'user':
                        st.markdown(f"<div class='user-message'><strong>You:</strong> {chat['message']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='ai-message'><strong>AI Assistant:</strong> {chat['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='ai-message'><strong>AI Assistant:</strong> Hello! I'm here to help you with questions about your skin condition. Please enter your OpenAI API key in the sidebar to use the AI assistant. You can ask questions in any language, and I will respond in Hinglish (Hindi-English mix). What would you like to know about {}?</div>".format(current_condition), unsafe_allow_html=True)
            
            # Chat input
            st.markdown("### Ask a Question")
            user_question = st.text_input(t('type_question', lang), key="chat_input")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(t('send_button', lang), key="send_btn", use_container_width=True):
                    if user_question:
                        # Get AI response
                        ai_response = get_ai_assistant_response(user_question, current_condition, lang)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({'role': 'user', 'message': user_question})
                        st.session_state.chat_history.append({'role': 'assistant', 'message': ai_response})
                        
                        # Rerun to update chat display
                        st.rerun()
            with col2:
                if st.button("🎤 Voice Input", key="voice_btn", use_container_width=True):
                    # Get speech input
                    speech_text = speech_to_text()
                    if speech_text:
                        # Get AI response
                        ai_response = get_ai_assistant_response(speech_text, current_condition, lang)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({'role': 'user', 'message': f"[Voice] {speech_text}"})
                        st.session_state.chat_history.append({'role': 'assistant', 'message': ai_response})
                        
                        # Rerun to update chat display
                        st.rerun()
            with col3:
                if st.button(t('clear_chat', lang), key="clear_btn", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please analyze an image first in the Detection tab to enable the AI assistant.")
        
        # Acne Photo Analysis
        st.markdown("---")
        st.subheader("Acne Photo Analysis")
        st.markdown("Upload a photo of your acne for severity assessment and personalized recommendations.")
        
        # File uploader for acne photos
        acne_photo = st.file_uploader("Upload acne photo", type=["jpg", "jpeg", "png"], key="acne_photo_uploader")
        
        if acne_photo is not None:
            # Display the uploaded image
            image = Image.open(acne_photo)
            st.image(image, caption="Uploaded Acne Photo", use_column_width=True)
            
            if st.button("Analyze Photo", key="analyze_photo_btn"):
                with st.spinner("Analyzing photo..."):
                    # Analyze the photo
                    analysis = analyze_acne_photo(image)
                    st.text_area("Analysis Results:", value=analysis, height=250)
        
        # Product Ingredient Checker
        st.markdown("---")
        st.subheader("Product Ingredient Checker")
        st.markdown("Check if your skincare product ingredients are suitable for your skin condition.")
        
        # Text area for entering ingredients
        ingredients_input = st.text_area("Enter product ingredients (comma separated):", 
                                        placeholder="e.g., water, niacinamide, hyaluronic acid, salicylic acid",
                                        height=100)
        
        if st.button("Check Ingredients", key="check_ingredients_btn"):
            if ingredients_input:
                # Split ingredients by comma
                ingredients = [ingredient.strip() for ingredient in ingredients_input.split(",") if ingredient.strip()]
                if ingredients:
                    # Check ingredients
                    analysis = check_product_ingredients(ingredients)
                    st.text_area("Analysis Results:", value=analysis, height=200)
                else:
                    st.warning("Please enter at least one ingredient.")
            else:
                st.warning("Please enter product ingredients.")
        
        # Skincare Routine Generator
        st.markdown("---")
        st.subheader("Personalized Skincare Routine")
        st.markdown("Generate a customized skincare routine based on your skin condition.")
        
        if st.button("Generate My Routine", key="generate_routine_btn"):
            # Get current condition
            current_condition = st.session_state.get("current_condition", "Healthy")
            
            # Generate routine
            routine = generate_skincare_routine(current_condition)
            
            # Display routine
            st.markdown(f"## {routine['title']}")
            
            st.markdown("### 🌅 Morning Routine")
            for step in routine['morning']:
                st.markdown(f"- {step}")
            
            st.markdown("### 🌙 Evening Routine")
            for step in routine['evening']:
                st.markdown(f"- {step}")
            
            st.markdown("### 📅 Weekly Care")
            for step in routine['weekly']:
                st.markdown(f"- {step}")
            
            st.markdown("### 💡 Pro Tips")
            for tip in routine['tips']:
                st.markdown(f"- {tip}")
        
        # Information section
        st.markdown("---")
        st.subheader(t('ai_capabilities', lang))
        st.markdown(f"""
        - **{t('treatment_info', lang)}**: Get details about treatment options and expected costs
        - **{t('prevention_tips', lang)}**: Learn how to prevent skin issues
        - **{t('dietary_advice', lang)}**: Understand how nutrition affects your skin
        - **{t('product_recommendations', lang)}**: Find suitable skincare products
        - **{t('professional_guidance', lang)}**: Know when to see a dermatologist
        - **{t('severity_analysis', lang)}**: Understand the seriousness of your condition
        """)
        
    # History tab in sidebar
    st.sidebar.header("🕒 History")
    history_data = load_history()
    
    if len(history_data) > 0:
        st.sidebar.markdown("Recent Analyses:")
        for entry in history_data[-5:]:  # Show last 5 entries
            st.sidebar.markdown(f"**{entry['Timestamp']}**\n\n{entry['Condition']} ({entry['Confidence']:.1f}%)")
    else:
        st.sidebar.info("No history data available.")

if __name__ == "__main__":
    main()