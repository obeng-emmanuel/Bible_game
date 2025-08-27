import os, json, io, re
from typing import List, Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import httpx

# Load SOP book metadata from JSON
with open("sop_books.json", "r", encoding="utf-8") as f:
    SOP_BOOKS_DATA = json.load(f)

SOP_BOOKS = list(SOP_BOOKS_DATA.keys())  # just the names for quick detection

# ---- LLM client (OpenAI-compatible) ----
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
MODEL = os.getenv("MODEL", "gpt-4o-mini")

app = FastAPI(title="AI Quiz Generator", version="1.0")

# ---------- Schemas ----------
class Source(BaseModel):
    mode: Literal["bible", "sop", "document", "mixed"]
    reference: Optional[str] = None

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

# ---------- Prompts ----------
BIBLE_SYSTEM_PROMPT = """You generate fair, respectful quiz questions from Bible passages.
- Avoid doctrinal slant; stick to the provided text.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["...","...","...","..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"bible","reference":"Book Chap:Vers (TRANSLATION)"},"explanation":"..."}]}
"""

DOC_SYSTEM_PROMPT = """You generate quiz questions from provided text (PDF/DOC/Images).
- Use only the given text; no outside info.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"document","reference":"<filename or section>"},"explanation":"..."}]}
"""

SOP_SYSTEM_PROMPT = """You generate quiz questions from Ellen White’s writings (Spirit of Prophecy).
- Respect the text and keep questions faithful.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"sop","reference":"Book name ch.#"},"explanation":"..."}]}
"""

MIXED_SYSTEM_PROMPT = """You generate quiz questions from BOTH Bible and Spirit of Prophecy texts.
- Some questions may come from Bible, some from SOP, some comparing both.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"mixed","reference":"Bible+SOP"},"explanation":"..."}]}
"""

# ---------- Helpers ----------
BOOKS = [ "Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges","Ruth",
"1 Samuel","2 Samuel","1 Kings","2 Kings","1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job",
"Psalms","Proverbs","Ecclesiastes","Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel",
"Hosea","Joel","Amos","Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai","Zechariah","Malachi",
"Matthew","Mark","Luke","John","Acts","Romans","1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians",
"Colossians","1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon","Hebrews","James",
"1 Peter","2 Peter","1 John","2 John","3 John","Jude","Revelation" ]

SOP_BOOKS = [
    "Steps to Christ","The Desire of Ages","The Great Controversy",
    "Patriarchs and Prophets","Prophets and Kings","Education","Christ’s Object Lessons"
]

REF_REGEX = re.compile(
    r"^\s*(?:[1-3]\s*)?[A-Za-z\. ]+\s+\d+(?::\d+(?:-\d+)?)?(?:\s*[,;]\s*\d+(?::\d+(?:-\d+)?)?)*\s*$"
)

def looks_like_bible_reference(text: str) -> bool:
    if not text: return False
    if len(text) > 80: return False
    return any(text.lower().startswith(b.lower()) for b in BOOKS) and bool(REF_REGEX.match(text))

async def fetch_bible_text(reference: str, translation: str = "KJV") -> Optional[str]:
    url = f"https://bible-api.com/{reference.replace(' ', '%20')}?translation={translation.lower()}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client_http:
            r = await client_http.get(url)
            if r.status_code != 200: return None
            data = r.json()
            if "text" in data and data["text"].strip(): return data["text"]
            if "verses" in data: return " ".join(v.get("text","") for v in data["verses"]).strip()
            return None
    except: return None

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
    if any(name.endswith(ext) for ext in [".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"]) or file.content_type.startswith("image/"):
        image = Image.open(io.BytesIO(raw))
        return pytesseract.image_to_string(image)
    try:
        return raw.decode("utf-8", errors="ignore")
    except: return ""

def normalize_items(items: List[dict]) -> List[Question]:
    cleaned = []
    for it in items:
        if "answer" not in it or "question" not in it or "source" not in it: continue
        if it.get("type","mcq") == "mcq" and (not it.get("choices") or len(it["choices"]) != 4): continue
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

@app.post("/api/generate/sop", response_model=List[Question])
def generate_sop(
    book: str = Form(...),
    chapter: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard")
):
    if book not in SOP_BOOKS:
        raise HTTPException(400, f"Unsupported SOP book: {book}")

    # Use provided text or build a placeholder reference
    passage = text or f"(Book: {book}, Chapter: {chapter or 'unknown'})"

    user_prompt = f"""
Generate {n} questions. Types: {types}.
Difficulty mix: {difficulty_mix}.
Source: {book} {chapter or ''}
Text:
\"\"\"\n{passage}\n\"\"\""""

    result = call_llm_json(SOP_SYSTEM_PROMPT, user_prompt)
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
        text=text, n=n,
        types=[t.strip() for t in types.split(",")],
        difficulty_mix=[d.strip() for d in difficulty_mix.split(",")],
        filename_or_ref=file.filename
    )
    return generate_from_text(req)

@app.post("/api/generate/mixed", response_model=List[Question])
def generate_mixed(
    bible_text: str = Form(...),
    sop_text: str = Form(...),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard")
):
    combined = f"Bible:\n{bible_text}\n\nSOP:\n{sop_text}"
    user_prompt = f"""
Generate {n} questions from BOTH Bible and SOP texts.
Label them as 'mixed' in source.
Text:
\"\"\"\n{combined}\n\"\"\""""
    result = call_llm_json(MIXED_SYSTEM_PROMPT, user_prompt)
    return normalize_items(result.get("items", []))

@app.post("/api/generate/auto", response_model=List[Question])
async def generate_auto(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    translation: Literal["KJV","WEB","Other"] = Form("KJV"),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard")
):
    allowed_types = [t.strip() for t in types.split(",") if t.strip()]
    diff_mix = [d.strip() for d in difficulty_mix.split(",") if d.strip()]

    if file is not None:
        extracted = extract_text_from_file(file)
        if not extracted.strip():
            raise HTTPException(400, "No text extracted from uploaded file")
        return generate_from_text(GenerateFromTextRequest(
            text=clamp_text(extracted), n=n,
            types=allowed_types, difficulty_mix=diff_mix,
            filename_or_ref=file.filename or "Uploaded file"
        ))

    if text and text.strip():
        t = text.strip()
        if looks_like_bible_reference(t):
            fetched = await fetch_bible_text(t, translation=translation)
            if fetched: return generate_bible(GenerateBibleRequest(
                translation=translation, reference=t, passage_text=fetched,
                n=n, types=allowed_types, difficulty_mix=diff_mix
            ))
        if any(t.lower().startswith(b.lower()) for b in SOP_BOOKS):
            return generate_sop(book=t, chapter=None, text=None, n=n, types=",".join(allowed_types), difficulty_mix=",".join(diff_mix))
        return generate_from_text(GenerateFromTextRequest(
            text=clamp_text(t), n=n,
            types=allowed_types, difficulty_mix=diff_mix,
            filename_or_ref="User input"
        ))

    raise HTTPException(400, "Provide either text or a file.")
