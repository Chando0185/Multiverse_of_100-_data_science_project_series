import os
import base64
import markdown
from markupsafe import Markup
from groq import Groq

# === CONFIGURATION ===
GROQ_API_KEY = ""  
IMAGE_PATH = "image.jpg" 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

# === Medical Analysis Prompt ===
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

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def analyze_image(image_path, groq_api_key):
    if not allowed_file(image_path):
        raise ValueError("Invalid file type. Allowed types: png, jpg, jpeg, dicom.")

    base64_image = encode_image(image_path)
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
    html_result = Markup(markdown.markdown(markdown_result, extensions=['fenced_code', 'tables']))
    
    return markdown_result, html_result

if __name__ == "__main__":
    try:
        markdown_text, html_text = analyze_image(IMAGE_PATH, GROQ_API_KEY)
        print("\n=== Analysis Result (Markdown) ===\n")
        print(markdown_text)
        
        print("\n=== Analysis Result (HTML) ===\n")
        print(html_text)
    except Exception as e:
        print(f"Error: {e}")
