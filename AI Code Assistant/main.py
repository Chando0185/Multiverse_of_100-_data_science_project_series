import os
from PIL import Image
import pytesseract
from agno.agent import Agent
from agno.models.groq import Groq
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
E2B_API_KEY = os.getenv("E2B_API_KEY")

code_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    markdown=True
)

def extract_code_from_markdown(markdown: str) -> str:
    if "```python" in markdown:
        return markdown.split("```python")[1].split("```")[0].strip()
    return markdown.strip()

def main():
    image_path = "test_question.png"  

    img = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(img).strip()
    if not extracted_text:
        print("❌ No text detected in image.")
        return

    print("=== Extracted Problem Description ===")
    print(extracted_text)
    print()

    code_prompt = f"""You're an expert Python developer. Write complete, optimal code for the problem below.

Problem:
{extracted_text}

Include type hints and sample input/output examples if possible."""

    response = code_agent.run(code_prompt)

    solution = extract_code_from_markdown(str(response))

    if not solution:
        print("❌ No valid code block found in Groq response.")
        return

    print("=== Generated Python Code ===")
    print(solution)
    print()

    os.environ["E2B_API_KEY"] = E2B_API_KEY
    sandbox = Sandbox(timeout=30)
    execution = sandbox.run_code(solution)

    if execution.error:
        error_explain = code_agent.run(f"Explain this Python error:\n{execution.error}")
        print("⚠️ Execution Error:")
        print(error_explain)
    else:
        output_lines = execution.logs.stdout if hasattr(execution.logs, "stdout") else str(execution.logs).splitlines()
        print("✅ Execution Output:")
        for line in output_lines:
            if line.strip():
                print(line.strip())

if __name__ == "__main__":
    main()
