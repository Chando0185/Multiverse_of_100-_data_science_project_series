import asyncio
from groq import Groq
import os
from dotenv import load_dotenv
import markdown

# Load environment variables from .env
load_dotenv()

def get_groq_client(api_key):
    return Groq(api_key=api_key)

async def ask_groq(prompt: str, api_key: str, model="llama-3.3-70b-versatile") -> str:
    try:
        client = get_groq_client(api_key)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI teaching assistant. Reply in Markdown format with well-structured sections and simulated Google Doc links."
                },
                {"role": "user", "content": prompt}
            ],
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

async def run_agents(topic, api_key):
    prompts = {
        "professor": f"Create a detailed knowledge base on '{topic}' including key concepts, applications, and fundamentals. Include a simulated Google Doc link.",
        "advisor": f"Design a structured learning roadmap for '{topic}', broken into beginner to expert levels. Include time estimates and a simulated Google Doc link.",
        "librarian": f"Curate a list of high-quality resources (videos, docs, blogs) for learning '{topic}' with descriptions. Simulate Google Doc link.",
        "assistant": f"Create practice exercises and real-world projects for mastering '{topic}', including solutions. Include a simulated Google Doc link."
    }
    tasks = [ask_groq(prompts[role], api_key) for role in prompts]
    raw_responses = await asyncio.gather(*tasks)

    responses = {}
    for role, markdown_text in zip(prompts.keys(), raw_responses):
        html = markdown.markdown(
            markdown_text,
            extensions=["fenced_code", "nl2br", "tables", "sane_lists"]
        )
        responses[role] = html
    return responses

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    topic = input("Enter the topic you want to learn about: ").strip()
    if not topic:
        print("Error: Topic cannot be empty.")
        return

    responses = asyncio.run(run_agents(topic, api_key))

    for role, content in responses.items():
        print(f"\n===== {role.upper()} =====\n")
        print(content)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
