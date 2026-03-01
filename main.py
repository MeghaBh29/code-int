import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# ─── FastAPI app setup ────────────────────────────────────────────────────────
app = FastAPI()

# CORS: allows the endpoint to be called from browsers / testing tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response models ────────────────────────────────────────────────
class CodeRequest(BaseModel):
    code: str                 # The Python code sent by the user

class CodeResponse(BaseModel):
    error: List[int]          # Line numbers where errors occurred (empty if none)
    result: str               # Exact stdout or full traceback string

# ─── Pydantic model for the AI's structured response ─────────────────────────
class ErrorAnalysis(BaseModel):
    error_lines: List[int]    # The line numbers the AI identifies as faulty

# ─── Tool Function ────────────────────────────────────────────────────────────
def execute_python_code(code: str) -> dict:
    """
    Runs the given Python code string.
    Captures whatever it prints (stdout) or any crash message (traceback).

    Returns:
        {"success": True,  "output": "printed text..."}   ← code ran fine
        {"success": False, "output": "Traceback ..."}     ← code crashed
    """
    # Redirect stdout so we can capture print() calls
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})                        # Run the code in a clean namespace
        output = sys.stdout.getvalue()        # Grab whatever was printed
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()       # Full crash message as a string
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout               # Always restore stdout!

# ─── AI Error Analysis ────────────────────────────────────────────────────────
def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Sends the broken code + its traceback to Gemini.
    Asks it to return ONLY the line number(s) where the error is.
    Uses structured output so the response is always clean JSON.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return the line number(s) where the error is located.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    # Parse the JSON response into our Pydantic model for safety
    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines

# ─── The Main Endpoint ────────────────────────────────────────────────────────
@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    """
    POST /code-interpreter
    Body: { "code": "..." }

    Flow:
      1. Run the code with execute_python_code()
      2. If it succeeded  → return {error: [], result: output}
      3. If it crashed    → ask AI which lines are broken
                         → return {error: [line_numbers], result: traceback}
    """
    # Step 1: Run the code
    execution = execute_python_code(request.code)

    # Step 2: No error → return immediately, no AI needed
    if execution["success"]:
        return CodeResponse(error=[], result=execution["output"])

    # Step 3: Error occurred → ask AI to analyze
    error_lines = analyze_error_with_ai(request.code, execution["output"])

    return CodeResponse(error=error_lines, result=execution["output"])
