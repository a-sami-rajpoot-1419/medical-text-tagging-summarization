from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mock data for sample texts
SAMPLE_DATA = {
    "Patient presents with severe chest pain and shortness of breath. The pain is described as a crushing sensation in the center of the chest, radiating to the left arm. Patient has a history of hypertension and type 2 diabetes. On examination, blood pressure is 160/95 mmHg, heart rate is 110 bpm, and respiratory rate is 24/min. ECG shows ST elevation in leads V1-V4. Patient was given aspirin 325mg and nitroglycerin sublingually. Immediate transfer to cardiac catheterization lab was arranged for possible percutaneous coronary intervention.": {
        "tags": ["Cardiology", "Acute Coronary Syndrome", "Hypertension", "Diabetes Mellitus", "Antiplatelet Therapy"],
        "summary": "Patient with history of hypertension and diabetes presents with acute chest pain and ECG changes consistent with myocardial infarction. Treated with aspirin and nitroglycerin, transferred for cardiac intervention.",
        "metrics": {
            "entity_recognition": 85,
            "summary_quality": 82,
            "medical_term_accuracy": 88,
            "context_understanding": 79
        }
    },
    "A 45-year-old male presents with persistent cough and fever for the past 5 days. Patient reports increasing fatigue and difficulty breathing. Temperature is 38.5°C, oxygen saturation is 92% on room air. Chest X-ray shows bilateral infiltrates. COVID-19 PCR test is positive. Patient is started on dexamethasone 6mg daily and supplemental oxygen. Antibiotics are initiated for possible secondary bacterial infection.": {
        "tags": ["Pulmonology", "COVID-19", "Pneumonia", "Respiratory Failure", "Corticosteroid Therapy"],
        "summary": "Middle-aged male with COVID-19 pneumonia showing respiratory symptoms and hypoxemia. Treated with dexamethasone and antibiotics, requiring supplemental oxygen.",
        "metrics": {
            "entity_recognition": 87,
            "summary_quality": 84,
            "medical_term_accuracy": 85,
            "context_understanding": 82
        }
    },
    "Patient reports severe headache and photophobia for the past 24 hours. History of migraines but current symptoms are more severe than usual. Neurological examination shows nuchal rigidity and positive Kernig's sign. Lumbar puncture reveals elevated white blood cell count and protein levels. Patient is started on intravenous ceftriaxone and vancomycin for suspected bacterial meningitis. MRI brain is ordered to rule out complications.": {
        "tags": ["Neurology", "Meningitis", "CNS Infection", "Antibiotic Therapy", "Neuroimaging"],
        "summary": "Patient with severe headache and meningeal signs suggestive of bacterial meningitis. CSF analysis shows inflammatory changes. Started on broad-spectrum antibiotics and imaging ordered.",
        "metrics": {
            "entity_recognition": 83,
            "summary_quality": 86,
            "medical_term_accuracy": 89,
            "context_understanding": 81
        }
    },
    "A 60-year-old female presents with progressive joint pain and morning stiffness lasting more than 2 hours. Physical examination reveals symmetrical swelling of proximal interphalangeal joints and metacarpophalangeal joints. Rheumatoid factor and anti-CCP antibodies are positive. X-rays show joint space narrowing and erosions. Patient is started on methotrexate 15mg weekly and folic acid supplementation. Regular monitoring of liver function tests is advised.": {
        "tags": ["Rheumatology", "Rheumatoid Arthritis", "Autoimmune Disease", "DMARD Therapy", "Joint Pathology"],
        "summary": "Elderly female with classic symptoms and serological markers of rheumatoid arthritis. Started on methotrexate with folic acid supplementation and monitoring plan.",
        "metrics": {
            "entity_recognition": 88,
            "summary_quality": 85,
            "medical_term_accuracy": 87,
            "context_understanding": 84
        }
    }
}

# Comment out model loading
"""
def load_models():
    # Tagging model
    tagging_tokenizer = AutoTokenizer.from_pretrained("models/tagging_finetuned")
    tagging_model = AutoModelForSequenceClassification.from_pretrained("models/tagging_finetuned")
    
    # Summarization model
    summarization_tokenizer = AutoTokenizer.from_pretrained("models/summarization_finetuned")
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("models/summarization_finetuned")
    
    return tagging_model, tagging_tokenizer, summarization_model, summarization_tokenizer

# Initialize models
tagging_model, tagging_tokenizer, summarization_model, summarization_tokenizer = load_models()
"""

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX files."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    
    elif file_ext == '.docx':
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from file
                text = extract_text_from_file(file_path)
                if text is None:
                    return jsonify({'error': 'Unsupported file format'})
                
                # Clean up
                os.remove(file_path)
        else:
            text = request.form.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'})
        
        # Check if the text matches any of our sample texts
        for sample_text, result in SAMPLE_DATA.items():
            if text.strip() == sample_text.strip():
                # Ensure all required fields are present and in correct format
                response_data = {
                    'tags': result['tags'],
                    'summary': result['summary'],
                    'metrics': {
                        'entity_recognition': int(result['metrics']['entity_recognition']),
                        'summary_quality': int(result['metrics']['summary_quality']),
                        'medical_term_accuracy': int(result['metrics']['medical_term_accuracy']),
                        'context_understanding': int(result['metrics']['context_understanding'])
                    }
                }
                print("Sending response:", response_data)  # Debug log
                return jsonify(response_data)
        
        # If no match found, return a default response
        return jsonify({
            'error': 'Text not recognized. Please use one of the sample texts provided.'
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Debug log
        return jsonify({
            'error': f'An error occurred while processing your request: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 