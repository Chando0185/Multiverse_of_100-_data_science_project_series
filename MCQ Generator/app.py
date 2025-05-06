import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize LangChain LLM
llm = ChatGroq(
    api_key="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",
    model="llama-3.3-70b-versatile",  # Updated model name
    temperature=0.0
)

# LangChain prompt template
mcq_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""
You are an AI assistant helping the user generate multiple-choice questions (MCQs) from the text below:

Text:
{context}

Generate {num_questions} MCQs. Each should include:
- A clear question
- Four answer options labeled A, B, C, and D
- The correct answer clearly indicated at the end

Format:
## MCQ
Question: [question]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [correct option]
"""
)

mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Text extraction
def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

# MCQ generation
def generate_mcqs_with_langchain(text, num_questions):
    response = mcq_chain.run({"context": text, "num_questions": num_questions})
    return response.strip()

# Save MCQs to text file
def save_mcqs_to_file(mcqs, filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    return path

# Save MCQs to PDF
def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)

    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    pdf.output(path)
    return path

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_file(file_path)
        if text:
            num_questions = int(request.form['num_questions'])
            mcqs = generate_mcqs_with_langchain(text, num_questions)

            # Save output
            base_name = filename.rsplit('.', 1)[0]
            txt_file = f"generated_mcqs_{base_name}.txt"
            pdf_file = f"generated_mcqs_{base_name}.pdf"
            save_mcqs_to_file(mcqs, txt_file)
            create_pdf(mcqs, pdf_file)

            return render_template('results.html', mcqs=mcqs, txt_filename=txt_file, pdf_filename=pdf_file)

    return "Invalid file format or upload error."

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
