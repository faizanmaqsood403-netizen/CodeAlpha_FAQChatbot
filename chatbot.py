"""FAQ chatbot engine: TF-IDF vectors + cosine similarity."""
import json
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FAQ_FILE = os.path.join(BASE_DIR, "faqs.json")

GREETINGS = {"hi", "hello", "hey", "salam", "assalamualaikum", "good morning",
             "good evening", "good afternoon"}
GOODBYES = {"bye", "goodbye", "exit", "quit", "see you", "thanks bye"}
THANKS = {"thanks", "thank you", "thank u", "thx", "shukriya"}


class FAQChatbot:
    def __init__(self, faq_file: str = DEFAULT_FAQ_FILE, threshold: float = 0.20):
        self.threshold = threshold
        self.faqs = self._load_faqs(faq_file)
        self.questions = [f["question"] for f in self.faqs]
        self.answers = [f["answer"] for f in self.faqs]

        # Preprocess every FAQ question and build the TF-IDF matrix.
        # The question is repeated so it weighs more than the answer text.
        self.processed_questions = [
            preprocess(q + " " + q + " " + a) for q, a in zip(self.questions, self.answers)
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.faq_matrix = self.vectorizer.fit_transform(self.processed_questions)

    @staticmethod
    def _load_faqs(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            raise ValueError("FAQ file is empty.")
        for item in data:
            if "question" not in item or "answer" not in item:
                raise ValueError("Every FAQ needs 'question' and 'answer' keys.")
        return data

    def get_best_match(self, user_text: str, top_n: int = 3):
        """Return (best_index or None, best_score, suggestions list)."""
        cleaned = preprocess(user_text)
        if not cleaned:
            return None, 0.0, []
        user_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, self.faq_matrix).flatten()
        ranked = scores.argsort()[::-1]
        best = int(ranked[0])
        best_score = float(scores[best])
        suggestions = [self.questions[i] for i in ranked[:top_n] if scores[i] > 0]
        if best_score < self.threshold:
            return None, best_score, suggestions
        return best, best_score, suggestions

    def get_response(self, user_text: str) -> dict:
        """Main entry point. Returns a dict with answer, matched question, score."""
        text = (user_text or "").strip()
        lowered = text.lower().strip("!.? ")

        if not text:
            return {"answer": "Please type a question so I can help you.",
                    "matched_question": None, "score": 0.0, "suggestions": [], "end": False}
        if lowered in GREETINGS:
            return {"answer": random.choice(["Hello! How can I help you today?",
                                             "Hi there! Ask me anything about our store."]),
                    "matched_question": None, "score": 1.0, "suggestions": [], "end": False}
        if lowered in THANKS:
            return {"answer": "You're welcome! Is there anything else I can help with?",
                    "matched_question": None, "score": 1.0, "suggestions": [], "end": False}
        if lowered in GOODBYES:
            return {"answer": "Goodbye! Have a great day.",
                    "matched_question": None, "score": 1.0, "suggestions": [], "end": True}

        idx, score, suggestions = self.get_best_match(text)
        if idx is None:
            return {"answer": "Sorry, I couldn't find an answer to that. "
                              "You can rephrase your question or contact support@example.com.",
                    "matched_question": None, "score": round(score, 3),
                    "suggestions": suggestions, "end": False}
        return {"answer": self.answers[idx], "matched_question": self.questions[idx],
                "score": round(score, 3), "suggestions": [], "end": False}
