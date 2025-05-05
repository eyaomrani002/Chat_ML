import os
import uuid
import logging
import bleach
import pandas as pd
import numpy as np
import nltk
import re
import string
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer, EnglishStemmer
from langdetect import detect, LangDetectException
from googletrans import Translator
import pdfplumber
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import faiss
from gtts import gTTS
from joblib import load
from transformers import BertTokenizer, BertForSequenceClassification, logging as transformers_logging
import torch
import speech_recognition as sr

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# Disable TensorFlow in transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Setup NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"NLTK download failed: {str(e)}")
    raise

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register Amiri font for PDF
amiri_font_path = 'Amiri-Regular.ttf'
if not os.path.exists(amiri_font_path):
    logging.error("Amiri-Regular.ttf not found. Download from https://fonts.google.com/specimen/Amiri")
    raise FileNotFoundError("Amiri-Regular.ttf not found.")
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('Amiri', amiri_font_path))
except Exception as e:
    logging.error(f"Font registration failed: {str(e)}")
    raise

# Initialize stopwords and stemmers
try:
    french_stopwords = stopwords.words('french')
    english_stopwords = stopwords.words('english')
    try:
        arabic_stopwords = stopwords.words('arabic')
    except LookupError:
        logging.warning("Arabic stopwords not available in NLTK. Using empty list.")
        arabic_stopwords = []
    french_stemmer = FrenchStemmer()
    english_stemmer = EnglishStemmer()
except Exception as e:
    logging.error(f"Error initializing NLTK resources: {str(e)}")
    raise

translator = Translator()

# Preprocessing function
def preprocess_text(text, lang='fr'):
    try:
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text, language='french' if lang == 'fr' else 'english' if lang == 'en' else 'arabic')
        stemmer = french_stemmer if lang == 'fr' else english_stemmer
        stopwords_list = french_stopwords if lang == 'fr' else english_stopwords if lang == 'en' else arabic_stopwords
        tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_list] if lang != 'ar' else [word for word in tokens if word not in stopwords_list]
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error in preprocess_text: {str(e)}")
        return ''

# Load dataset
try:
    df = pd.read_csv('iset_questions_reponses.csv')
    logging.info("Dataset loaded successfully")
except FileNotFoundError:
    logging.error("iset_questions_reponses.csv not found.")
    raise FileNotFoundError("iset_questions_reponses.csv not found.")

# Load ratings
ratings = pd.read_csv('ratings.csv') if os.path.exists('ratings.csv') else pd.DataFrame(columns=['response_id', 'rating', 'timestamp'])

# Load models
try:
    vectorizer = load('tfidf_vectorizer.joblib')
    nb_model = load('nb_model.joblib')
    svm_model = load('svm_model.joblib')
    tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Catégorie'].unique()))
    model.load_state_dict(torch.load('bert_model.pt'))
    model.eval()
    logging.info("Models and tokenizer loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {str(e)}")
    raise FileNotFoundError("Run the Jupyter notebook to generate models.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Preprocess dataset
try:
    df['Processed_Question'] = df['Question'].apply(lambda x: preprocess_text(x, 'fr'))
    X = vectorizer.transform(df['Processed_Question']).toarray().astype(np.float32)
    logging.info("Dataset preprocessed successfully")
except Exception as e:
    logging.error(f"Error preprocessing dataset: {str(e)}")
    raise

# Load or create FAISS index
if not os.path.exists('faiss_index.bin'):
    logging.info("FAISS index not found. Creating a new index...")
    dimension = X.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(X)
    faiss.write_index(index, 'faiss_index.bin')
else:
    index = faiss.read_index('faiss_index.bin')
logging.info("FAISS index loaded or created")

# Dictionnaires de questions-réponses
qa_fr = {
    "comment puis-je m'inscrire ?": "Vous pouvez vous inscrire via le portail étudiant.",
    "quels sont les cours offerts ?": "Consultez le programme sur le site.",
    "où trouver le calendrier académique ?": "Le calendrier est sur le portail étudiant.",
    "comment contacter le support technique ?": "Envoyez un email à support@iset.tn."
}
qa_en = {
    "how can I register?": "You can register through the student portal.",
    "what courses are offered?": "Check the program on the website.",
    "where can I find the academic calendar?": "The calendar is on the student portal.",
    "how to contact technical support?": "Send an email to support@iset.tn."
}
qa_ar = {
    "كيف يمكنني التسجيل؟": "يمكنك التسجيل من خلال بوابة الطلاب.",
    "ما هي الدورات المقدمة؟": "تحقق من البرنامج على الموقع.",
    "أين يمكنني العثور على التقويم الأكاديمي؟": "التقويم موجود على بوابة الطلاب.",
    "كيفية التواصل مع الدعم الفني؟": "أرسل بريدًا إلكترونيًا إلى support@iset.tn."
}

NO_RESPONSE_MESSAGES = {
    'fr': "Aucune réponse détectée.",
    'en': "No response detected.",
    'ar': "لم يتم الكشف عن أي إجابة."
}

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        logging.warning(f"pdfplumber failed: {str(e)}")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logging.error(f"PyPDF2 failed: {str(e)}")
    return text

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='fra+eng+ara')
        return text
    except Exception as e:
        logging.error(f"Image text extraction failed: {str(e)}")
        return ""

def classify_question_bert(question):
    try:
        inputs = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        category_id = logits.argmax(dim=1).item()
        category = df['Catégorie'].unique()[category_id]
        confidence = torch.softmax(logits, dim=1)[0, category_id].item()
        return {'category': category, 'confidence': confidence}
    except Exception as e:
        logging.error(f"Error in classify_question_bert: {str(e)}")
        return {'category': 'Général', 'confidence': 0.0}

def get_best_response(user_input, lang='fr'):
    try:
        processed_input = preprocess_text(user_input, lang)
        input_vec = vectorizer.transform([processed_input]).toarray().astype(np.float32)

        # FAISS similarity
        distances, indices = index.search(input_vec, 1)
        idx = indices[0][0]
        dataset_similarity = 1 - (distances[0][0] / 2)

        # Cosine similarity with qa_fr/qa_en/qa_ar
        data = qa_fr if lang == 'fr' else qa_en if lang == 'en' else qa_ar
        questions = list(data.keys())
        processed_questions = [preprocess_text(q, lang=lang) for q in questions]
        vectorizer_temp = TfidfVectorizer()
        X_temp = vectorizer_temp.fit_transform(processed_questions + [processed_input])
        similarities = cosine_similarity(X_temp[-1], X_temp[:-1])
        max_idx = similarities.argmax()
        qa_similarity = similarities[0, max_idx]

        # Classification with BERT
        bert_result = classify_question_bert(processed_input)
        category = bert_result['category']
        confidence = bert_result['confidence']

        # Choose best response
        if qa_similarity > dataset_similarity and qa_similarity >= 0.1:
            answer = data[questions[max_idx]]
            link = ''
            source = 'qa_dict'
        else:
            answer = df.iloc[idx]['Réponse']
            link = df.iloc[idx].get('Lien', '')
            source = 'dataset'

        # Translate if needed
        if lang != 'fr' and source == 'dataset':
            try:
                answer = translator.translate(answer, dest=lang).text
                category = translator.translate(category, dest=lang).text
            except Exception as e:
                logging.warning(f"Translation failed: {str(e)}")

        response = {
            'answer': answer,
            'link': link,
            'category': category,
            'response_id': str(uuid.uuid4()),
            'confidence': float(confidence),
            'similarity': max(float(dataset_similarity), float(qa_similarity)),
            'ask_for_response': max(dataset_similarity, qa_similarity) < 0.1
        }
        logging.info(f"Response generated: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in get_best_response: {str(e)}")
        return {
            'answer': NO_RESPONSE_MESSAGES.get(lang, NO_RESPONSE_MESSAGES['fr']),
            'link': '',
            'category': 'Général',
            'response_id': str(uuid.uuid4()),
            'confidence': 0.0,
            'similarity': 0.0,
            'ask_for_response': True
        }

def sanitize_input(text):
    return bleach.clean(text, tags=[], strip=True)

@app.route('/')
def home():
    try:
        logging.info("Rendering chat.html")
        return render_template('chat.html')
    except Exception as e:
        logging.error(f"Error rendering chat.html: {str(e)}")
        return jsonify({'error': 'Template not found. Ensure chat.html is in templates/'}), 500

@app.route('/chat', methods=['POST'])
def chat_handler():
    try:
        logging.info("Received /chat request")
        # Validate and process files
        pdf_text = ""
        image_text = ""
        if 'pdf_file' in request.files:
            pdf = request.files['pdf_file']
            if pdf.filename:
                if not pdf.filename.endswith('.pdf'):
                    logging.warning("Invalid PDF file")
                    return jsonify({'error': 'Seuls les fichiers PDF sont acceptés.'}), 400
                filename = f"{uuid.uuid4()}.pdf"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf.save(pdf_path)
                pdf_text = extract_text_from_pdf(pdf_path)
                logging.info(f"PDF processed: {filename}")

        if 'image_file' in request.files:
            image = request.files['image_file']
            if image.filename:
                if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logging.warning("Invalid image file")
                    return jsonify({'error': 'Seuls les fichiers PNG/JPEG sont acceptés.'}), 400
                filename = f"{uuid.uuid4()}{os.path.splitext(image.filename)[1]}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                image_text = extract_text_from_image(image_path)
                logging.info(f"Image processed: {filename}")

        # Process question
        question = sanitize_input(request.form.get('message', ''))
        output_lang = request.form.get('output_lang', 'fr')
        use_voice = request.form.get('use_voice', 'false') == 'true'
        logging.info(f"Question: {question}, Language: {output_lang}, Voice: {use_voice}")

        # Detect input language
        try:
            input_lang = detect(question) if question else 'fr'
        except LangDetectException:
            input_lang = 'fr'
        logging.info(f"Detected input language: {input_lang}")

        # Combine question with PDF and image text
        full_input = f"{question} {pdf_text} {image_text}".strip()
        if not full_input:
            logging.warning("No valid input provided")
            return jsonify({'error': 'Aucun message ou fichier valide fourni.'}), 400

        response = get_best_response(full_input, output_lang)

        # Generate audio if voice mode is enabled
        if use_voice:
            try:
                tts = gTTS(text=response['answer'], lang=output_lang)
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                tts.save(audio_path)
                response['audio'] = f"/{audio_path}"
                logging.info(f"Audio generated: {audio_filename}")
            except Exception as e:
                logging.warning(f"Failed to generate audio: {str(e)}")
                response['audio'] = None

        logging.info(f"Returning response: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in chat_handler: {str(e)}")
        return jsonify({'error': f"Erreur serveur: {str(e)}. Veuillez réessayer."}), 500

@app.route('/add_response', methods=['POST'])
def add_response():
    try:
        logging.info("Received /add_response request")
        data = request.json
        question = sanitize_input(data['question'])
        response = sanitize_input(data['response'])
        link = sanitize_input(data.get('link', ''))
        category = sanitize_input(data.get('category', 'Général'))

        new_row = pd.DataFrame([{
            'Question': question,
            'Réponse': response,
            'Lien': link,
            'Catégorie': category,
            'Rating': 0
        }])

        global df, X, index
        df = pd.concat([df, new_row], ignore_index=True)
        df['Processed_Question'] = df['Question'].apply(lambda x: preprocess_text(x, 'fr'))
        X = vectorizer.transform(df['Processed_Question']).toarray().astype(np.float32)
        index.reset()
        index.add(X)
        faiss.write_index(index, 'faiss_index.bin')
        df.to_csv('iset_questions_reponses.csv', index=False)
        logging.info("New response added to dataset")

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error in add_response: {str(e)}")
        return jsonify({'error': f"Erreur lors de l'ajout de la réponse: {str(e)}"}), 500

@app.route('/rate', methods=['POST'])
def rate_response():
    try:
        logging.info("Received /rate request")
        data = request.json
        response_id = sanitize_input(data['response_id'])
        rating = sanitize_input(data['rating'])

        global ratings
        new_rating = pd.DataFrame([{
            'response_id': response_id,
            'rating': rating,
            'timestamp': pd.Timestamp.now()
        }])
        ratings = pd.concat([ratings, new_rating], ignore_index=True)
        ratings.to_csv('ratings.csv', index=False)
        logging.info(f"Rating added: {response_id}, {rating}")

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error in rate_response: {str(e)}")
        return jsonify({'error': f"Erreur lors de l'évaluation: {str(e)}"}), 500

@app.route('/export_conversations', methods=['POST'])
def export_conversations():
    try:
        logging.info("Received /export_conversations request")
        data = request.json
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont('Amiri', 12)

        y = 750
        for conv in data['conversations']:
            question = sanitize_input(conv['question'])
            answer = sanitize_input(conv['answer'])
            p.drawString(50, y, f"Q: {question[:100]}...")
            y -= 20
            p.drawString(50, y, f"R: {answer[:100]}...")
            y -= 30
            if y < 100:
                p.showPage()
                p.setFont('Amiri', 12)
                y = 750

        p.save()
        buffer.seek(0)
        logging.info("Conversations exported to PDF")
        return send_file(buffer, mimetype='application/pdf', download_name='conversation.pdf')
    except Exception as e:
        logging.error(f"Error in export_conversations: {str(e)}")
        return jsonify({'error': f"Erreur lors de l'exportation PDF: {str(e)}"}), 500

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    try:
        logging.info(f"Serving uploaded file: {filename}")
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        logging.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': f"Erreur lors de l'accès au fichier: {str(e)}"}), 500

if __name__ == '__main__':
    print("Flask app initialized successfully. Ready to serve requests.")
    app.run(debug=False)