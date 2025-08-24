import os
from uuid import uuid4
from pathlib import Path
from flask import Flask, render_template, request, flash
from agno.agent import Agent
from agno.models.groq import Groq
from agno.media import Image as AgnoImage
import markdown2

app = Flask(__name__)
app.secret_key = "recovery-app-secret"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------
# Markdown renderer for Jinja2
# ------------------------------
@app.template_filter("markdown")
def markdown_filter(text):
    if not text:
        return ""
    return markdown2.markdown(text, extras=["fenced-code-blocks", "tables", "strike", "underline"])


# ------------------------------
# Create Agents
# ------------------------------
def create_agents(api_key: str):
    model = Groq(id="llama-3.3-70b-versatile", api_key=api_key)

    therapist = Agent(
        model=model,
        name="Therapist Agent",
        instructions=[
            "You are an empathetic breakup therapist.",
            "Respond using Markdown with clear **bold numbered headings** and bullet points.",
            "Example:\n"
            "### Coping With Loss\n"
            "**1. Validate Feelings:**\n- Show empathy.\n"
            "**2. Encourage:**\n- Share hopeful messages."
        ],
        markdown=True,
    )

    closure = Agent(
        model=model,
        name="Closure Message Agent",
        instructions=[
            "You help people express unspoken feelings after a breakup.",
            "Respond using Markdown with bold numbered headers and short bullet points.",
            "Example:\n"
            "### Closure\n"
            "**1. Unsent Messages:**\n- Write down feelings.\n"
            "**2. Emotional Release:**\n- Try journaling."
        ],
        markdown=True,
    )

    planner = Agent(
        model=model,
        name="Recovery Planner Agent",
        instructions=[
            "You are a breakup recovery planner.",
            "Provide a **7-day recovery plan** using Markdown with bold day headings and bullet points.",
            "Example:\n"
            "### 7-Day Plan\n"
            "**Day 1: Self-care**\n- Take a walk\n- Meditate"
        ],
        markdown=True,
    )

    honest_agent = Agent(
        model=model,
        name="Brutal Honesty Agent",
        instructions=[
            "You are a no-nonsense breakup advisor.",
            "Give tough-love advice using Markdown with bold numbered headings and bullets.",
            "Example:\n"
            "### Brutal Honesty\n"
            "**1. Face Reality:**\n- Accept what went wrong.\n"
            "**2. Move On:**\n- Focus on growth."
        ],
        markdown=True,
    )

    return therapist, closure, planner, honest_agent


# ------------------------------
# Flask Route
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    therapist_response = closure_response = planner_response = honest_response = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        files = request.files.getlist("images")

        # Load Groq API key from environment only
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            flash("❌ Please set the GROQ_API_KEY environment variable.", "error")
            return render_template("index.html")

        if not user_input and not files:
            flash("⚠️ Please provide your story or upload images.", "error")
            return render_template("index.html")

        # Convert uploaded files to AgnoImage list
        agno_images = []
        for file in files:
            if file.filename:
                filepath = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")
                file.save(filepath)
                agno_images.append(AgnoImage(filepath=Path(filepath)))

        try:
            # Initialize agents
            therapist, closure, planner, honest = create_agents(groq_key)

            # Run all agents
            therapist_response = therapist.run(
                f"The user is feeling: {user_input}", images=agno_images
            ).content
            closure_response = closure.run(
                f"Help them express unsaid thoughts: {user_input}", images=agno_images
            ).content
            planner_response = planner.run(
                f"Design a 7-day recovery plan for: {user_input}", images=agno_images
            ).content
            honest_response = honest.run(
                f"Give brutal advice about this situation: {user_input}", images=agno_images
            ).content

        except Exception as e:
            flash(f"❌ Something went wrong: {str(e)}", "error")

    return render_template(
        "index.html",
        therapist=therapist_response,
        closure=closure_response,
        planner=planner_response,
        honesty=honest_response,
    )


if __name__ == "__main__":
    app.run(debug=True)
