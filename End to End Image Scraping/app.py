from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import logging
import pymongo
import os

logging.basicConfig(filename="scrapper.log", level=logging.INFO)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def homepage():
    return render_template("index.html", message=None)

@app.route("/review", methods=['POST'])
def index():
    try:
        # query to search for images
        query = request.form['content'].replace(" ", "")

        # directory to store images
        save_directory = "images/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        # Google Images request
        response = requests.get(
            f"https://www.google.com/search?q={query}&tbm=isch",
            headers=headers
        )

        soup = BeautifulSoup(response.text, "html.parser")
        image_tags = soup.find_all("img")

        if len(image_tags) > 1:
            del image_tags[0]  # remove google logo

        img_data = []

        for index, image_tag in enumerate(image_tags[:20]):  # limit to faster downloads
            image_url = image_tag.get("src")
            if image_url:
                try:
                    img_content = requests.get(image_url).content
                    file_path = os.path.join(save_directory, f"{query}_{index}.jpg")

                    with open(file_path, "wb") as f:
                        f.write(img_content)

                    img_data.append({"Index": index, "Image_URL": image_url})
                except:
                    continue

        # MongoDB Store (only metadata)
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["image_scrap"]
        review_col = db["image_scrap_data"]
        review_col.insert_many(img_data)

        return render_template(
            "index.html",
            message=f"✔ Successfully saved {len(img_data)} images for '{query}'!"
        )

    except Exception as e:
        logging.error(str(e))
        return render_template("index.html", message="❌ Something went wrong. Try again!")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
