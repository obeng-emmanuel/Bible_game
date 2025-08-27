# main.py
import os, json, io, re
from typing import List, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import Response
from pydantic import BaseModel

from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from PIL import Image

# OCR guard (avoid crashing if Tesseract binary isn't installed)
try:
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

import httpx

from text_source import LocalJsonSource
from generator import generate_questions_from_text


app = FastAPI(title="AI Quiz Generator", version="1.0")


# ---------- App data locations ----------
DATA_DIR = os.getenv("DATA_DIR", "data")
BIBLE_DIR = os.getenv("BIBLE_DIR", os.path.join(DATA_DIR, "bible", "kjv"))
EGW_DIR   = os.getenv("EGW_DIR",   os.path.join(DATA_DIR, "egw"))
ENABLE_REMOTE_BIBLE = os.getenv("ENABLE_REMOTE_BIBLE", "false").lower() == "true"


@app.on_event("startup")
def startup():
    # Load local Bible + SOP JSONs once
    app.state.source = LocalJsonSource(BIBLE_DIR, EGW_DIR)


# ---------- Root & health ----------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/health")
def health(verbose: bool = Query(False)):
    bible_list = app.state.source.bible_books()
    sop_list = app.state.source.sop_books()
    resp = {
        "ok": True,
        "bible_books": len(bible_list),
        "sop_books": len(sop_list),
    }
    if verbose:
        resp["bible_loaded"] = bible_list
        resp["sop_loaded"] = sop_list
    return JSONResponse(resp)

@app.head("/health")
def health_head():
    return Response(status_code=200)


# ---------- Optional listings (handy for Swagger testing) ----------
@app.get("/books", response_model=List[str])
def list_bible_books():
    return app.state.source.bible_books()

@app.get("/chapters/{book}")
def bible_chapter_count(book: str):
    try:
        return {"book": book, "chapters": app.state.source.bible_chapter_count(book)}
    except KeyError:
        raise HTTPException(404, f"Bible book not found: {book}")

@app.get("/sop/books", response_model=List[str])
def list_sop_books():
    return app.state.source.sop_books()

@app.get("/sop/chapters/{title}")
def sop_chapters(title: str):
    if title not in app.state.source._egw_index:
        raise HTTPException(404, f"SOP book not found: {title}")
    return {"title": title, "chapters": len(app.state.source._egw_index[title])}


# ---------- LLM client (OpenAI-compatible) ----------
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
MODEL = os.getenv("MODEL", "gpt-4o-mini")


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
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["...","...","...","..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"document","reference":"<filename or section>"},"explanation":"..."}]}
"""

SOP_SYSTEM_PROMPT = """You generate quiz questions from Ellen White’s writings (Spirit of Prophecy).
- Respect the text and keep questions faithful.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["...","...","...","..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"sop","reference":"Book name ch.#"},"explanation":"..."}]}
"""

MIXED_SYSTEM_PROMPT = """You generate quiz questions from BOTH Bible and Spirit of Prophecy texts.
- Some questions may come from Bible, some from SOP, some comparing both.
- Return ONLY JSON in this shape:
{"items":[{"type":"mcq|true_false|short_answer","question":"...","choices":["...","...","...","..."],"answer":"...","difficulty":"easy|medium|hard","source":{"mode":"mixed","reference":"Bible+SOP"},"explanation":"..."}]}
"""


# ---------- Helpers ----------
REF_REGEX = re.compile(
    r"^\s*(?:[1-3]\s*)?[A-Za-z\. ]+\s+\d+(?::\d+(?:-\d+)?)?(?:\s*[,;]\s*\d+(?::\d+(?:-\d+)?)?)*\s*$"
)

def looks_like_bible_reference(text: str) -> bool:
    if not text:
        return False
    if len(text) > 80:
        return False
    # light heuristic: starts with a loaded Bible book name
    return any(text.lower().startswith(b.lower()) for b in app.state.source.bible_books()) and bool(REF_REGEX.match(text))

async def fetch_bible_text(reference: str, translation: str = "KJV") -> Optional[str]:
    url = f"https://bible-api.com/{reference.replace(' ', '%20')}?translation={translation.lower()}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client_http:
            r = await client_http.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            if "text" in data and data["text"].strip():
                return data["text"]
            if "verses" in data:
                return " ".join(v.get("text", "") for v in data["verses"]).strip()
            return None
    except Exception:
        return None

from openai import OpenAI

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model = os.getenv("MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)

def clamp_text(text: str, max_chars: int = 15000) -> str:
    return (text or "").strip()[:max_chars]

def extract_text_from_file(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    raw = file.file.read()

    if name.endswith(".pdf"):
        return pdf_extract_text(io.BytesIO(raw))

    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    if any(name.endswith(ext) for ext in [".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"]) or (file.content_type or "").startswith("image/"):
        if not HAS_OCR:
            raise HTTPException(400, "OCR not enabled on this server (install tesseract-ocr).")
        image = Image.open(io.BytesIO(raw))
        return pytesseract.image_to_string(image)

    # fallback: try plain text
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def normalize_items(items: List[dict]) -> List[Question]:
    cleaned: List[Question] = []
    for it in items or []:
        if "answer" not in it or "question" not in it or "source" not in it:
            continue
        if it.get("type", "mcq") == "mcq":
            ch = (it.get("choices") or [])
            if len(ch) < 2:
                continue
            # pad/truncate to 4 so the UI stays consistent
            it["choices"] = (ch + ["N/A","N/A","N/A","N/A"])[:4]
        cleaned.append(Question(
            type=it.get("type", "mcq"),
            question=(it["question"] or "").strip(),
            choices=it.get("choices"),
            answer=(it["answer"] or "").strip(),
            difficulty=it.get("difficulty", "medium"),
            source=Source(**it["source"]),
            explanation=it.get("explanation")
        ))
    return cleaned


# ---------- Generation endpoints ----------
@app.post("/api/generate/bible", response_model=List[Question])
async def generate_bible(req: GenerateBibleRequest):
    # (1) direct text provided
    if req.passage_text and req.passage_text.strip():
        text = clamp_text(req.passage_text)

    # (2) try local JSONs by reference
    elif req.reference:
        try:
            text = app.state.source.passage_text(req.reference)
        except Exception:
            text = None

        # (3) optional fallback to remote API
        if not text and ENABLE_REMOTE_BIBLE:
            fetched = await fetch_bible_text(req.reference, translation=req.translation)
            if not fetched:
                raise HTTPException(400, f"Could not resolve reference: {req.reference}")
            text = clamp_text(fetched)

    else:
        raise HTTPException(422, "Provide 'reference' or 'passage_text'.")

    result = call_llm_json(
        BIBLE_SYSTEM_PROMPT,
        f"""Generate {req.n} questions. Types: {", ".join(req.types)}.
Difficulty mix: {", ".join(req.difficulty_mix)}.
Translation: {req.translation}.
Passage:
"""\n{text}\n""""""
    )
    return normalize_items(result.get("items", []))


@app.post("/api/generate/sop", response_model=List[Question])
def generate_sop(
    book: str = Form(...),
    chapter: Optional[int] = Form(None),
    text: Optional[str] = Form(None),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard"),
):
    # choose text source: direct text OR (book,chapter)
    if text and text.strip():
        passage = clamp_text(text)
        ref = f"{book} (custom text)"
    else:
        if not chapter:
            raise HTTPException(422, "Provide 'chapter' when no text is supplied.")
        try:
            passage = clamp_text(app.state.source.sop_chapter_text(book, int(chapter)))
            ref = f"{book} ch.{chapter}"
        except Exception as e:
            raise HTTPException(400, str(e))

    result = call_llm_json(
        SOP_SYSTEM_PROMPT,
        f"""Generate {n} questions. Types: {types}.
Difficulty mix: {difficulty_mix}.
Source: {ref}
Text:
"""\n{passage}\n""""""
    )
    return normalize_items(result.get("items", []))


@app.post("/api/generate/from-text", response_model=List[Question])
def generate_from_text(req: GenerateFromTextRequest):
    text = clamp_text(req.text)
    ref = req.filename_or_ref or "Provided text"
    result = call_llm_json(
        DOC_SYSTEM_PROMPT,
        f"""Generate {req.n} questions. Types: {", ".join(req.types)}.
Difficulty mix: {", ".join(req.difficulty_mix)}.
Source text:
"""\n{text}\n"""\n
Reference: {ref}
"""
    )
    return normalize_items(result.get("items", []))


@app.post("/api/generate/from-upload", response_model=List[Question])
def generate_from_upload(
    file: UploadFile = File(...),
    n: int = Form(5),
    types: str = Form("mcq"),
    difficulty_mix: str = Form("easy,medium,hard"),
):
    text = extract_text_from_file(file)
    if not text.strip():
        raise HTTPException(400, "No text extracted from file")
    req = GenerateFromTextRequest(
        text=text, n=n,
        types=[t.strip() for t in types.split(",") if t.strip()],
        difficulty_mix=[d.strip() for d in difficulty_mix.split(",") if d.strip()],
        filename_or_ref=file.filename,
    )
    return generate_from_text(req)


@app.post("/api/generate/mixed", response_model=List[Question])
def generate_mixed(
    bible_text: str = Form(...),
    sop_text: str = Form(...),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard"),
):
    combined = f"Bible:\n{bible_text}\n\nSOP:\n{sop_text}"
    result = call_llm_json(
        MIXED_SYSTEM_PROMPT,
        f"""Generate {n} questions from BOTH Bible and SOP texts.
Label them as 'mixed' in source.
Text:
"""\n{combined}\n""""""
    )
    return normalize_items(result.get("items", []))


@app.post("/api/generate/auto", response_model=List[Question])
async def generate_auto(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    translation: Literal["KJV","WEB","Other"] = Form("KJV"),
    n: int = Form(5),
    types: str = Form("mcq,true_false"),
    difficulty_mix: str = Form("easy,medium,hard"),
):
    allowed_types = [t.strip() for t in types.split(",") if t.strip()]
    diff_mix = [d.strip() for d in difficulty_mix.split(",") if d.strip()]

    # File upload path
    if file is not None:
        extracted = extract_text_from_file(file)
        if not extracted.strip():
            raise HTTPException(400, "No text extracted from uploaded file")
        return generate_from_text(GenerateFromTextRequest(
            text=clamp_text(extracted), n=n,
            types=allowed_types, difficulty_mix=diff_mix,
            filename_or_ref=file.filename or "Uploaded file"
        ))

    # Raw text path
    if text and text.strip():
        t = text.strip()
        if looks_like_bible_reference(t):
            local_text = None
            try:
                local_text = app.state.source.passage_text(t)
            except Exception:
                local_text = None
            if not local_text and ENABLE_REMOTE_BIBLE:
                fetched = await fetch_bible_text(t, translation=translation)
                local_text = fetched
            if local_text:
                return await generate_bible(GenerateBibleRequest(
                    translation=translation, reference=t, passage_text=local_text,
                    n=n, types=allowed_types, difficulty_mix=diff_mix
                ))
        # SOP quick guess: starts with a known SOP title
        if any(t.lower().startswith(b.lower()) for b in app.state.source.sop_books()):
            # If chapter not in text, require user to call /api/generate/sop explicitly.
            return generate_from_text(GenerateFromTextRequest(
                text=clamp_text(t), n=n,
                types=allowed_types, difficulty_mix=diff_mix,
                filename_or_ref="User input (SOP-like)"
            ))
        # Fallback: plain text
        return generate_from_text(GenerateFromTextRequest(
            text=clamp_text(t), n=n,
            types=allowed_types, difficulty_mix=diff_mix,
            filename_or_ref="User input"
        ))

    raise HTTPException(400, "Provide either text or a file.")
