import os
import time
from groq import Groq
from gtts import gTTS

groq_client = Groq(api_key="")

def transcribe_audio(filepath):
    with open(filepath, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
        )
    return response.text

def get_answer(question):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful agriculture chatbot for Indian farmers."},
            {"role": "user", "content": "Give a Brief Of Agriculture Seasons in India"},
            {"role": "system", "content": "In India, the agricultural season consists of three major seasons: the Kharif (monsoon), the Rabi (winter), and the Zaid (summer)..."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

def typing_effect(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Newline at end

def text_to_speech(text, filename):
    tts = gTTS(text)
    output_path = f"{filename}.mp3"
    tts.save(output_path)
    return output_path

def main():
    mode = input("Choose input type ('text' or 'audio'): ").strip().lower()

    if mode == 'text':
        question = input("Enter your question: ").strip()

    elif mode == 'audio':
        filepath = input("Enter the path to your audio file: ").strip()
        if not os.path.exists(filepath):
            print("❌ File not found.")
            return
        print("🎤 Transcribing audio...")
        question = transcribe_audio(filepath)
        print(f"📝 Transcribed Text: {question}")

    else:
        print("❌ Invalid input type. Use 'text' or 'audio'.")
        return

    print("🤖 Getting response from LLM...")
    answer = get_answer(question)

    print("\n✅ Answer:")
    typing_effect(answer) 

    print("\n🔊 Converting answer to speech...")
    audio_file = text_to_speech(answer, "response_audio")
    print(f"🎧 Voice saved to: {audio_file}")

if __name__ == "__main__":
    main()
