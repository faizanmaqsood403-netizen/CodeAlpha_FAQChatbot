"""Flask web UI for the FAQ chatbot."""
from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))
    return jsonify(bot.get_response(message))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
