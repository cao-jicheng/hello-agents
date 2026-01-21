import os
from dotenv import load_dotenv

load_dotenv()

exact_search = os.getenv("QDRANT_EXACT_SEARCH").lower()

print(type(exact_search))
print(exact_search == "true")