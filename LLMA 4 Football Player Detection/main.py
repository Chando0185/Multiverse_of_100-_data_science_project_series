import cv2
import requests
import base64
import numpy as np
import os


GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def detect_faces(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, img_rgb

    largest_face = max(faces, key = lambda r : r[2] * r[3])
    return [largest_face], img_rgb

def draw_faces(image, faces, player_name):
    for x, y, w, h, in faces:
        cv2.rectangle(image, (x, y), (x + w, y+h), (0, 255, 0), 3)
        font_scale = max(0.5, min(1.5, 30/len(player_name)))
        cv2.putText(image, player_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return image

def get_player_info(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":"application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are an expert AI trained in recognizing professional football (soccer) players with high accuracy. 
                                Carefully analyze the given image and identify the player with certainty. 
                                If a well-known player is present, return the following details in a structured format:

                                - **Full Name**: (Only the player's full name, no extra text)
                                - **Nationality**: (Country of origin)
                                - **Club**: (Current club name)
                                - **Position**: (Primary playing position)
                                - **Achievements**: (Notable titles or awards, max 2-3)

                                If multiple players are detected, return only the **most famous one**.
                                If no player is recognized, respond with "Unknown".
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(GROQ_API_URL, headers = headers, json = payload)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print("API Error", response.text)
        return "UNKNOWN"

def main():
    image_path = "neymar.jpg"
    if not os.path.exists(image_path):
        print(f"Image Not Found: {image_path}")
        return
    
    faces, img_rgb = detect_faces(image_path)
    if faces:
        player_info = get_player_info(image_path)
        print(player_info)

        player_name = player_info.split("\n")[0].replace("**Full Name**: ", "").strip()
        result_img = draw_faces(img_rgb, faces, player_name)

        out_path = "output_detected.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"Output saved to {out_path}")
    else:
        print("No faces detected . Try another image.")

if __name__=="__main__":
    main()