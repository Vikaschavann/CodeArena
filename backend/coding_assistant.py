import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class CodingAssistant:
    def __init__(self, problem_context: str, language: str):
        self.problem_context = problem_context
        self.language = language
        
        self.system_instruction = f"""You are a senior competitive programming mentor on CodeArena.
Problem Context:
{problem_context}

Programming Language: {language}

STRICT BEHAVIOR:
- Do NOT directly rewrite the entire solution unless explicitly asked.
- Prefer debugging hints over full answers. Tell the user *where* to look.
- Explain time complexity.
- Mention edge cases.
- If error is runtime → explain memory/null issue.
- If TLE → suggest optimization.
- If WA → suggest edge case testing.
- Keep your answers concise, formatted nicely in Markdown, and friendly."""

        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            # In the new SDK, we use chat sessions with history
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config=genai.types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.4
                )
            )

    def predict(self, input_text: str) -> str:
        if not GEMINI_API_KEY:
            return "Error: Gemini API key is missing. Please configure it in the .env file."
        try:
            response = self.chat.send_message(input_text)
            if not response or not response.text:
                return "AI returned an empty response. This can happen if the content was flagged by safety filters. Please try rephrasing."
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                return "The AI service is currently busy due to high demand. Please wait a minute and try again."
            return f"Error communicating with AI: {error_msg}"

def create_coding_assistant(problem_context: str, language: str):
    return CodingAssistant(problem_context, language)
