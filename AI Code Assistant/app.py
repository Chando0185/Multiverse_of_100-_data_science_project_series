import os
from flask import Flask, request, render_template
from PIL import Image
import pytesseract
from agno.agent import Agent
from agno.models.groq import Groq
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
load_dotenv()

app = Flask(__name__)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
E2B_API_KEY = os.getenv("E2B_API_KEY")

code_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    markdown=True
)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted = None
    solution = None
    result = None

    if request.method == "POST":
        query = request.form.get("query")
        image = request.files.get("image")

        if image and not query:
            img = Image.open(image.stream)
            extracted_text = pytesseract.image_to_string(img).strip()
            extracted = extracted_text if extracted_text else "❌ No text detected in image."

            if extracted and extracted != "❌ No text detected in image.":
                code_prompt = f"""You're an expert Python developer. Write complete, optimal code for the problem below.

Problem:
{extracted}

Include type hints and sample input/output examples if possible."""
                response = code_agent.run(code_prompt)
                solution = extract_code_from_markdown(response.content)

                if solution:
                    os.environ["E2B_API_KEY"] = E2B_API_KEY
                    sandbox = Sandbox(timeout=30)
                    execution = sandbox.run_code(solution)

                    if execution.error:
                        error_explain = code_agent.run(f"Explain this Python error:\n{execution.error}")
                        result = f"""<div class='bg-red-100 text-red-800 p-4 rounded'>
                            <strong>⚠️ Error:</strong><br><pre>{error_explain.content}</pre>
                        </div>"""
                    else:
                        output_lines = execution.logs.stdout if hasattr(execution.logs, "stdout") else str(execution.logs).splitlines()
                        formatted_output = "<br>".join(line.strip() for line in output_lines if line.strip())

                        result = f"""<div class='bg-green-100 text-green-900 p-4 rounded'>
                            <strong>✅ Output:</strong><br><pre>{formatted_output}</pre>
                        </div>"""
                else:
                    result = "<div class='bg-yellow-100 p-3 rounded'>❌ No valid code block found in Groq response.</div>"

            return render_template("index.html", extracted=extracted, solution=solution, result=result)

        elif query and not image:
            code_prompt = f"""You're an expert Python developer. Write complete, optimal code for the problem below.

Problem:
{query}

Include type hints and sample input/output examples if possible."""
            response = code_agent.run(code_prompt)
            solution = extract_code_from_markdown(response.content)

            if solution:
                os.environ["E2B_API_KEY"] = E2B_API_KEY
                sandbox = Sandbox(timeout=30)
                execution = sandbox.run_code(solution)

                if execution.error:
                    error_explain = code_agent.run(f"Explain this Python error:\n{execution.error}")
                    result = f"""<div class='bg-red-100 text-red-800 p-4 rounded'>
                        <strong>⚠️ Error:</strong><br><pre>{error_explain.content}</pre>
                    </div>"""
                else:
                    output_lines = execution.logs.stdout if hasattr(execution.logs, "stdout") else str(execution.logs).splitlines()
                    formatted_output = "<br>".join(line.strip() for line in output_lines if line.strip())

                    result = f"""<div class='bg-green-100 text-green-900 p-4 rounded'>
                        <strong>✅ Output:</strong><br><pre>{formatted_output}</pre>
                    </div>"""
            else:
                result = "<div class='bg-yellow-100 p-3 rounded'>❌ No valid code block found in Groq response.</div>"
        else:
            extracted = "⚠️ Please provide either a query or image (not both)."

    return render_template("index.html", extracted=extracted, solution=solution, result=result)


def extract_code_from_markdown(markdown: str) -> str:
    if "```python" in markdown:
        return markdown.split("```python")[1].split("```")[0].strip()
    return markdown.strip()


if __name__ == "__main__":
    app.run(debug=True)
