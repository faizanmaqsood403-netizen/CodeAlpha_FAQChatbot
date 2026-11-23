# CodeAlpha_FAQ_Chatbot

AI internship — **Task 2: Chatbot for FAQs** (CodeAlpha)

A chatbot that answers customer questions by matching them with the most similar FAQ.

## How it works
1. **Collect FAQs** – `faqs.json` (25 question/answer pairs for an online store; replace with any topic).
2. **Preprocess** – `preprocess.py` uses **NLTK**: lower-casing, punctuation removal, tokenizing, stop-word removal, synonym mapping and Porter stemming.
3. **Match** – `chatbot.py` converts FAQs to **TF-IDF** vectors and finds the best FAQ using **cosine similarity** (scikit-learn).
4. **Respond** – the best answer is returned. If the similarity is below the threshold (0.20), the bot says it didn't understand and suggests related questions.
5. **Chat UI** (optional) – Flask web interface (`app.py` + `templates/index.html`) and a terminal version (`cli.py`).

## Project structure
```
CodeAlpha_FAQ_Chatbot/
├── app.py              # Flask web UI server
├── chatbot.py          # TF-IDF + cosine similarity engine
├── cli.py              # Command-line chatbot
├── preprocess.py       # NLTK text preprocessing
├── faqs.json           # FAQ dataset
├── test_chatbot.py     # Quick self-test
├── requirements.txt
└── templates/index.html
```

## Setup & Run
```bash
pip install -r requirements.txt

python app.py        # web UI -> open http://127.0.0.1:5000
python cli.py        # or terminal chat
python test_chatbot.py   # self-test
```
(NLTK stop-words are downloaded automatically on first run; if offline, a built-in list is used.)

## Use your own FAQs
Edit `faqs.json` and keep the format:
```json
{"question": "Your question?", "answer": "Your answer."}
```

## Tech
Python 3.8+, NLTK, scikit-learn, Flask, HTML/CSS/JS
