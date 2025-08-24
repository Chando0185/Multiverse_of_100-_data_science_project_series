import os
from uuid import uuid4
from pathlib import Path
from agno.agent import Agent
from agno.models.groq import Groq
from agno.media import Image as AgnoImage


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


os.environ["GROQ_API_KEY"] = ""

def create_agents(api_key: str):
    model = Groq(id="llama-3.3-70b-versatile", api_key=api_key)

    therapist = Agent(
        model=model,
        name="Therapist Agent",
        instructions=[
            "You are an empathetic breakup therapist.",
            "Respond using clear, bold numbered headings and bullet points to organize your advice.",
            "For example:\n"
            "**1. Validate Feelings:**\n- Show empathy and understanding.\n"
            "**2. Encourage:**\n- Share hopeful messages."
        ],
        markdown=True
    )

    closure = Agent(
        model=model,
        name="Closure Message Agent",
        instructions=[
            "You help people express unspoken feelings after a breakup.",
            "Respond in bullet points with bold numbered headers, making each step or idea clear.",
            "Example:\n"
            "**1. Unsent Messages:**\n- Write down your feelings honestly.\n"
            "**2. Emotional Release:**\n- Use journaling or meditation."
        ],
        markdown=True
    )

    planner = Agent(
        model=model,
        name="Recovery Planner Agent",
        instructions=[
            "You are a breakup recovery planner.",
            "Provide a 7-day recovery plan using numbered bold headings and bullet points for tasks and activities.",
            "Example:\n"
            "**Day 1: Self-care**\n- Take a walk\n- Practice mindfulness"
        ],
        markdown=True
    )

    honest_agent = Agent(
        model=model,
        name="Brutal Honesty Agent",
        instructions=[
            "You are a no-nonsense breakup advisor.",
            "Give honest feedback in bullet points with bold numbered headings.",
            "Example:\n"
            "**1. Face Reality:**\n- Accept what went wrong.\n"
            "**2. Move On:**\n- Focus on personal growth."
        ],
        markdown=True
    )

    return therapist, closure, planner, honest_agent


def main():
    # Load Groq API key from environment variable
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        print("❌ Please set your GROQ_API_KEY environment variable first.")
        return

    user_input = input("\nDescribe your breakup situation: ").strip()
    image_paths = input("\n(Optional) Enter image paths separated by commas (or press Enter to skip): ").strip()

    # Load images if provided
    agno_images = []
    if image_paths:
        for path in image_paths.split(","):
            path = path.strip()
            if os.path.exists(path):
                new_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{os.path.basename(path)}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(path, "rb") as src, open(new_path, "wb") as dst:
                    dst.write(src.read())
                agno_images.append(AgnoImage(filepath=Path(new_path)))

    try:
        # Initialize agents
        therapist, closure, planner, honest = create_agents(groq_key)

        print("\n======================")
        print("💙 Therapist Response:")
        print("======================")
        print(therapist.run(f"The user is feeling: {user_input}", images=agno_images).content)

        print("\n======================")
        print("💌 Closure Response:")
        print("======================")
        print(closure.run(f"Help them express unsaid thoughts: {user_input}", images=agno_images).content)

        print("\n======================")
        print("📅 Recovery Planner Response:")
        print("======================")
        print(planner.run(f"Design a recovery plan for: {user_input}", images=agno_images).content)

        print("\n======================")
        print("⚡ Brutal Honesty Response:")
        print("======================")
        print(honest.run(f"Give brutal advice about this situation: {user_input}", images=agno_images).content)

    except Exception as e:
        print(f"\n❌ Something went wrong: {str(e)}")


if __name__ == "__main__":
    main()
