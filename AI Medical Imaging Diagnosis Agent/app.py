import os
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from groq import Groq
import markdown
from markupsafe import Markup

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Your medical analysis prompt
MEDICAL_QUERY = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
def index():
    result_html = None
    error = None
    groq_api_key = ""
    
    if request.method == "POST":
        groq_api_key = request.form.get("groq_api_key", "").strip()
        if not groq_api_key:
            error = "Groq API Key is required."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key)

        if "image" not in request.files:
            error = "No image file part."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key)
        
        file = request.files["image"]
        if file.filename == "":
            error = "No selected file."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                base64_image = encode_image(filepath)
                client = Groq(api_key=groq_api_key)
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": MEDICAL_QUERY},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                )
                markdown_result = chat_completion.choices[0].message.content
                
                # Convert markdown to HTML
                result_html = Markup(markdown.markdown(markdown_result, extensions=['fenced_code', 'tables']))
            
            except Exception as e:
                error = f"Error during analysis: {e}"
        else:
            error = "Allowed image types are png, jpg, jpeg, dicom."

    return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key)

if __name__ == "__main__":
    app.run(debug=True)
