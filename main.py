import os, json, io
from typing import List, Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# ---- LLM client (OpenAI-compatible) ----
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
MODEL = os.getenv("MODEL", "gpt-4o-mini")

app = FastAPI(title="AI Quiz Generator", version="0.1")

# ---------- Schemas ----------
class Source(BaseModel):
    mode: Literal["bible", "document"]
    reference: Optional[str] = None   # e.g., "John 3:16 (KJV)" or "myfile.pdf p.3-5"

class Question(BaseModel):
    type: Literal["mcq", "true_false", "short_answer"]
    question: str
    choices: Optional[List[str]] = None
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    source: Source
    explanation: Optional[str] = None

class GenerateBibleRequest(BaseModel):
    translation: Literal["KJV", "WEB", "Other"] = "KJV"
    reference: Optional[str] = None
    passage_text: Optional[str] = None
    n: int = 5
    types: List[Literal["mcq", "true_false", "short_answer"]] = ["mcq"]
    difficulty_mix: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"]

class GenerateFromTextRequest(BaseModel):
    text: str
    n: int = 5
    types: List[Literal["mcq", "true_false", "short_answer"]] = ["mcq"]
    difficulty_mix: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"]
    filename_or_ref: Optional[str] = None

# ---------- Utilities ----------
BIBLE_SYSTEM_PROMPT = """You generate fair, respectful quiz questions from Bible passages.
Return ONLY JSON in this format:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"bible","reference":"Book Chap:Vers (TRANSLATION)"},"explanation":"..."},...]}
"""

DOC_SYSTEM_PROMPT = """You generate quiz questions from provided text (PDF/DOC/Images).
Return ONLY JSON in this format:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"document","reference":"<filename>"},"explanation":"..."},...]}
"""

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)

def clamp_text(text: str, max_chars: int = 15000) -> str:
    return text.strip()[:max_chars]

def extract_text_from_file(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    raw = file.file.read()
    if name.endswith(".pdf"):
        return pdf_extract_text(io.BytesIO(raw))
    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join([p.text for p in doc.paragraphs])
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]) or file.content_type.startswith("image/"):
        image = Image.open(io.BytesIO(raw))
        return pytesseract.image_to_string(image)
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def normalize_items(items: List[dict]) -> List[Question]:
    cleaned = []
    for it in items:
        if "answer" not in it or "question" not in it or "source" not in it:
            continue
        cleaned.append(Question(
            type=it.get("type","mcq"),
            question=it["question"].strip(),
            choices=it.get("choices"),
            answer=it["answer"].strip(),
            difficulty=it.get("difficulty","medium"),
            source=Source(**it["source"]),
            explanation=it.get("explanation")
        ))
    return cleaned

# ---------- Endpoints ----------
@app.post("/api/generate/bible", response_model=List[Question])
def generate_bible(req: GenerateBibleRequest):
    passage = req.passage_text or f"(Reference: {req.reference})"
    passage = clamp_text(passage)
    user_prompt = f"""
Generate {req.n} questions. Types: {", ".join(req.types)}.
Difficulty mix: {", ".join(req.difficulty_mix)}.
Translation: {req.translation}.
Passage:
\"\"\"\n{passage}\n\"\"\""""
    result = call_llm_json(BIBLE_SYSTEM_PROMPT, user_prompt)
    return normalize_items(result.get("items", []))

@app.post("/api/generate/from-text", response_model=List[Question])
def generate_from_text(req: GenerateFromTextRequest):
    text = clamp_text(req.text)
    ref = req.filename_or_ref or "Provided text"
    user_prompt = f"""
Generate {req.n} questions. Types: {", ".join(req.types)}.
Difficulty mix: {", ".join(req.difficulty_mix)}.
Source text:
\"\"\"\n{text}\n\"\"\"\n
Reference: {ref}
"""
    result = call_llm_json(DOC_SYSTEM_PROMPT, user_prompt)
    return normalize_items(result.get("items", []))

@app.post("/api/generate/from-upload", response_model=List[Question])
async def generate_from_upload(
    file: UploadFile = File(...),
    n: int = Form(5),
    types: str = Form("mcq"),
    difficulty_mix: str = Form("easy,medium,hard")
):
    text = extract_text_from_file(file)
    if not text.strip():
        raise HTTPException(400, "No text extracted from file")
    req = GenerateFromTextRequest(
        text=text,
        n=n,
        types=[t.strip() for t in types.split(",")],
        difficulty_mix=[d.strip() for d in difficulty_mix.split(",")],
        filename_or_ref=file.filename
    )
    return generate_from_text(req)

