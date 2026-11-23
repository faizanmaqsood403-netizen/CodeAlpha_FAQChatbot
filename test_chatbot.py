"""Quick self-test: python test_chatbot.py"""
from chatbot import FAQChatbot

bot = FAQChatbot()
tests = [
    ("how do i make a new account", "How can I create an account?"),
    ("forgot my password", "I forgot my password. How do I reset it?"),
    ("which payment options do you have", "What payment methods do you accept?"),
    ("when will my parcel arrive", "How long does delivery take? When will my order arrive?"),
    ("i want my money back", None),
    ("can i send items back", "How do I return a product?"),
    ("track my order", "How can I track my order?"),
    ("do you have an android app", "Do you have a mobile app?"),
]
passed = 0
for q, expected in tests:
    r = bot.get_response(q)
    ok = expected is None or r["matched_question"] == expected
    passed += ok
    print(("PASS" if ok else "FAIL"), "|", q, "->", r["matched_question"], r["score"])
print(f"\n{passed}/{len(tests)} passed")
