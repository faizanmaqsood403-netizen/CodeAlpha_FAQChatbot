"""Command-line version of the FAQ chatbot."""
from chatbot import FAQChatbot


def main():
    bot = FAQChatbot()
    print("=" * 60)
    print(" FAQ Chatbot  (type 'exit' or 'bye' to quit)")
    print("=" * 60)
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        reply = bot.get_response(user)
        print(f"Bot: {reply['answer']}")
        if reply["suggestions"]:
            print("     Did you mean:")
            for s in reply["suggestions"]:
                print(f"       - {s}")
        if reply["end"]:
            break


if __name__ == "__main__":
    main()
