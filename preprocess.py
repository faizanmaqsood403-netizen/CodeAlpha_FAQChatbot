"""Text preprocessing utilities using NLTK (tokenize, clean, stop-word removal, stemming)."""
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Small built-in fallback list, used only if NLTK stop-words cannot be downloaded.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does",
    "did", "i", "me", "my", "you", "your", "we", "our", "it", "its", "of", "to",
    "in", "on", "at", "for", "with", "and", "or", "if", "can", "could", "will",
    "would", "should", "how", "what", "when", "where", "which", "who", "this",
    "that", "there", "have", "has", "had", "so", "as", "by", "from", "about",
}

# Words that carry meaning in FAQs even though NLTK treats them as stop-words.
_KEEP_WORDS = {"not", "no", "cancel"}


def _load_stopwords():
    """Return the English stop-word set (downloads NLTK data once if needed)."""
    try:
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:  # no internet / any NLTK problem -> use fallback
        words = set(_FALLBACK_STOPWORDS)
    return words - _KEEP_WORDS


# Phrase / word synonyms so that different wording maps to the same FAQ vocabulary.
_PHRASE_SYNONYMS = {
    "money back": "refund",
    "send back": "return",
    "give back": "return",
    "when will": "how long",
    "how soon": "how long",
    "sign up": "create account",
    "signup": "create account",
    "log in": "login",
    "sign in": "login",
}
_WORD_SYNONYMS = {
    "make": "create", "register": "create", "new": "create",
    "parcel": "order", "package": "order", "purchase": "order", "buy": "order",
    "arrive": "delivery", "arrives": "delivery", "deliver": "delivery",
    "shipping": "delivery", "shipment": "delivery",
    "cost": "charges", "price": "charges", "fee": "charges", "fees": "charges",
    "pay": "payment", "paying": "payment", "options": "methods", "option": "methods",
    "android": "mobile", "iphone": "mobile", "ios": "mobile", "phone": "mobile",
    "application": "app", "email": "email", "mail": "email",
    "broken": "damaged", "defective": "damaged", "faulty": "damaged",
    "reset": "reset", "recover": "reset", "forgot": "forgot", "forget": "forgot",
    "coupon": "discount", "voucher": "discount", "promo": "discount", "code": "code",
    "contact": "contact", "reach": "contact", "call": "contact", "helpline": "contact",
    "remove": "delete", "close": "delete", "erase": "delete",
    "edit": "update", "change": "update", "modify": "update",
    "secure": "safe", "security": "safe", "privacy": "safe",
}

_STOP_WORDS = _load_stopwords()
_TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")  # NLTK tokenizer, needs no extra data
_STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    """Lower-case, remove punctuation, tokenize, drop stop-words and stem.

    Returns the cleaned text as a single space-separated string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove punctuation/symbols
    text = re.sub(r"\b(send|give|get|take|sending)\b(\s+\w+)?\s+back\b", "return", text)
    for phrase, repl in _PHRASE_SYNONYMS.items():      # phrase synonyms
        text = text.replace(phrase, repl)
    tokens = _TOKENIZER.tokenize(text)                 # tokenize
    tokens = [_WORD_SYNONYMS.get(t, t) for t in tokens]  # word synonyms
    tokens = [t for t in tokens if t not in _STOP_WORDS]  # remove stop-words
    tokens = [_STEMMER.stem(t) for t in tokens]        # stemming
    return " ".join(tokens)
