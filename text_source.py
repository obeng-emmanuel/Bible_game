# app/text_source.py
import os, json, re, hashlib, time
from typing import Dict, List, Optional

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

class LocalJsonSource:
    """
    Reads Bible/SOP from your local JSON data folders.
    - Bible: { "book": "Mark", "chapters": [["v1","v2",...], ["v1","v2",...], ...] }
    - SOP:   { "title": "Steps to Christ", "chapters": [{ "title":"...","text":"..."}, ...] }
    """
    def __init__(self, bible_dir: str, egw_dir: Optional[str] = None):
        self.bible_dir = bible_dir
        self.egw_dir = egw_dir
        self._bible_index: Dict[str, List[List[str]]] = {}
        self._egw_index: Dict[str, List[Dict[str,str]]] = {}
        self._load_all()

    def _load_all(self):
        if os.path.isdir(self.bible_dir):
            for fn in os.listdir(self.bible_dir):
                if not fn.lower().endswith(".json"): 
                    continue
                with open(os.path.join(self.bible_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                book = data.get("book") or os.path.splitext(fn)[0]
                self._bible_index[book] = data["chapters"]
        if self.egw_dir and os.path.isdir(self.egw_dir):
            for fn in os.listdir(self.egw_dir):
                if not fn.lower().endswith(".json"): 
                    continue
                with open(os.path.join(self.egw_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("title") or os.path.splitext(fn)[0].replace("_"," ")
                self._egw_index[title] = data.get("chapters", [])

    # --- Introspection
    def bible_books(self) -> List[str]:
        return sorted(self._bible_index.keys())

    def bible_chapter_count(self, book: str) -> int:
        return len(self._bible_index[book])

    # --- Text access
    def passage_text(self, reference: str) -> str:
        # "Mark 5" or "John 3:16-18"
        m = re.match(r'^\s*([1-3]?\s*[A-Za-z ]+)\s*(\d+)?(?:\:(\d+)(?:-(\d+))?)?\s*$', reference)
        if not m:
            raise ValueError(f"Unrecognized reference: {reference}")
        book = re.sub(r'\s+',' ', m.group(1).strip()).title()
        ch = int(m.group(2) or 0)
        v1 = int(m.group(3) or 0)
        v2 = int(m.group(4) or v1 or 0)
        if book not in self._bible_index:
            raise ValueError(f"Book not loaded locally: {book}")
        if ch <= 0:
            raise ValueError("Please provide a chapter, e.g. 'Mark 5' or 'John 3:16-18'")
        chapters = self._bible_index[book]
        if ch > len(chapters):
            raise ValueError(f"{book} has only {len(chapters)} chapters.")
        verses = chapters[ch-1]
        if v1 <= 0:  # whole chapter
            return " ".join(verses)
        if v2 <= 0: v2 = v1
        if v1 > len(verses) or v2 > len(verses):
            raise ValueError(f"{book} {ch} has only {len(verses)} verses.")
        return " ".join(verses[v1-1:v2])

# Optional remote source if you want internet fallback
class RemoteBibleSource:
    def __init__(self, cache_dir: str, get_url):
        import requests  # only needed if you enable this
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.get_url = get_url  # function(book,ch,v1,v2)->str

    def passage_text(self, reference: str) -> str:
        import requests
        # TODO: parse reference same as LocalJsonSource (can reuse that regex)
        # then build URL with self.get_url(book, ch, v1, v2)
        # fetch and cache result
        raise NotImplementedError("Fill in with chosen API if you want internet support.")
