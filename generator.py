# generator.py
import re, random
from typing import List

STOPWORDS = set("the a an and or of to in that is was for on with as by from at be this".split())

def _pick_keywords(words: List[str], k=3):
    cands = [w for w in words if w.isalpha() and w.lower() not in STOPWORDS and len(w) > 4]
    random.shuffle(cands)
    return cands[:k] or (words[:1] if words else [])

def make_mcq(sentence: str):
    words = re.findall(r"\w+", sentence)
    if not words:
        return None
    answer = _pick_keywords(words, 1)[0]
    blanked = re.sub(rf"\b{re.escape(answer)}\b", "____", sentence, count=1)
    # distractors from sentence
    pool = list({w for w in words if w.lower() != answer.lower() and len(w) >= 4})
    random.shuffle(pool)
    options = [answer] + pool[:3]
    random.shuffle(options)
    return {"type": "mcq", "question": blanked, "choices": options, "answer": answer}

def make_tf(sentence: str):
    q = sentence
    ans = "True"
    if re.search(r"\b(is|are|was|were|has|have)\b", sentence, re.I) and random.random() < 0.5:
        q = re.sub(r"\bis\b", "is not", sentence, flags=re.I, count=1)
        q = re.sub(r"\bare\b", "are not", q, flags=re.I, count=1)
        ans = "False"
    return {"type": "true_false", "question": q, "answer": ans}

def make_short(sentence: str):
    words = re.findall(r"\w+", sentence)
    if not words:
        return None
    key = _pick_keywords(words, 1)[0]
    q = re.sub(rf"\b{re.escape(key)}\b", "____", sentence, count=1)
    return {"type": "short_answer", "question": q, "answer": key}

def generate_questions_from_text(text: str, n=5, types=("mcq",), difficulty=("easy","medium","hard")):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if len(s.split()) >= 6] or [text.strip()]
    makers = {"mcq": make_mcq, "true_false": make_tf, "short_answer": make_short}
    out = []
    while len(out) < n:
        s = random.choice(sents)
        t = random.choice(list(types))
        q = makers[t](s)
        if q:
            q["difficulty"] = random.choice(list(difficulty))
            out.append(q)
    return out
