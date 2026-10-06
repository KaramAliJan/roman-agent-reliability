import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("API key not found")