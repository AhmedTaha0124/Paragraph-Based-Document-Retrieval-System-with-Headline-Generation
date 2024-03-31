# app.py

import spacy
from flask import Flask, render_template, request
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import torch
from docx import Document

nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)

# Load SpaCy model for NLP cleaning
nlp = spacy.load("en_core_web_sm")

# Load model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("AhmedTaha012/pargraphs_titlesV1.1")

totalPargraghs=[]

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text.strip()

def clean_paragraph(paragraph):
    # Tokenize into sentences
    sentences = sent_tokenize(paragraph)

    # Clean each sentence
    cleaned_sentences = [clean_text(sentence) for sentence in sentences]

    # Join the sentences back into a paragraph
    cleaned_paragraph = ' '.join(cleaned_sentences)

    return cleaned_paragraph

def clean_and_process():
    # Implement your NLP cleaning process using SpaCy
    cleanedPargraghs=[]
    for par in totalPargraghs:
        cleanedPargraghs.append(clean_paragraph(par))
    return cleanedPargraghs

def splitDocumentsToPargraghs(document):
    ## split document to pargaphs
    global totalPargraghs 
    totalPargraghs=totalPargraghs+[elm for elm in document.split("\n\n") if len(elm)>5]


# Function to generate embeddings using the sequence-to-sequence model
def generate_embeddings(sentence):
    # Tokenize the input text
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids

    # Forward pass to get embeddings
    with torch.no_grad():
        outputs = model(input_ids,decoder_input_ids=input_ids)
    return  outputs["encoder_last_hidden_state"].mean(dim=1).squeeze().numpy()
# app.py
def prediction(text):
    preds=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(model.generate(inputs=tokenizer(text,return_tensors="pt")["input_ids"])[0].tolist(),skip_special_tokens=True))
    return preds

def handleDiffrentFilesFormats(uploaded_files):
    content_list = []
    for uploaded_file in uploaded_files:
        if uploaded_file.filename != '':
            if uploaded_file.filename.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
            elif uploaded_file.filename.endswith('.docx'):
                doc = Document(uploaded_file)
                content = '\n\n'.join([paragraph.text for paragraph in doc.paragraphs])
            else:
                raise "Unsupported file format. Please upload .txt or .docx files."
            content_list.append(content)
    return content_list











@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    topic = request.form['topic']
    documents =  request.files.getlist('documents[]')
    documentsContent=handleDiffrentFilesFormats(documents)
   
    # Process each document
    for document in documentsContent:
        splitDocumentsToPargraghs(document)

    cleaned_Pargraghs=clean_and_process()    
    sentences=[topic]+cleaned_Pargraghs
    sentence_embeddings = [generate_embeddings(sentence) for sentence in sentences]
    similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
    similar_sentence_index = similarity_matrix[0].argsort()[-2]
    selectedPargragh=sentences[similar_sentence_index]
    title=prediction(selectedPargragh)
    return render_template('index.html', result=selectedPargragh,generated_title=title,similarity_score=str(similarity_matrix[0][similar_sentence_index]*100)+"%",cleaned_Pargraghs=cleaned_Pargraghs, generated_titles=[prediction(x) for x in cleaned_Pargraghs ])
    
if __name__ == '__main__':
    app.run(debug=True)