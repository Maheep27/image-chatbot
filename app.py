from src.apps import chat_page_new
from src.utils.MultiPage import MultiPage



app = MultiPage()

app.add_app("Chat", chat_page_new.app)

app.run()

