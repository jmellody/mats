from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os

cl = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

ids = sorted(m.id for m in cl.models.list())
print("--- gemma ---")
for i in ids:
    if "gemma" in i:
        print(i)
print("--- flash-lite ---")
for i in ids:
    if "flash-lite" in i or "flash_lite" in i:
        print(i)