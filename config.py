from dotenv import load_dotenv
import os
import instructor
from openai import OpenAI

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

def get_instructor_client():
    if LLM_PROVIDER == "groq":
        return instructor.from_openai(
            OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            ),
            mode=instructor.Mode.JSON
        ), "llama-3.1-8b-instant"
    # else:
    #     return instructor.from_openai(
    #         OpenAI(
    #             api_key="ollama",
    #             base_url="http://localhost:11434/v1"
    #         ),
    #         mode=instructor.Mode.JSON
    #     ), "qwen2.5:7b"