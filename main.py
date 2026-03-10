import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ValidationError, Field

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
load_dotenv()

TRVLR_API_BASE = os.getenv("TRVLR_API_BASE", "").rstrip("/")
TRVLR_API_TOKEN = os.getenv("TRVLR_API_TOKEN", "")
TRVLR_ORG_ID = os.getenv("TRVLR_ORG_ID", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MASTER_PROMPT_PATH = os.getenv("MASTER_PROMPT_PATH", "prompts/master_prompt_v1.txt")

# Step 4: guardrails / controls (recommended)
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
OPENAI_TIMEOUT_SECS = int(os.getenv("OPENAI_TIMEOUT_SECS", "45"))
OPENAI_RETRIES = int(os.getenv("OPENAI_RETRIES", "2"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "50"))

# -------------------------------------------------
# MASTER PROMPT LOADER
# -------------------------------------------------
def load_master_prompt() -> str:
    try:
        with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"Master prompt file not found: {MASTER_PROMPT_PATH}")

# -------------------------------------------------
# STEP 3: Helpers + Batch request model
# -------------------------------------------------
def clamp_text(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit]

class BatchRequest(BaseModel):
    pks: List[int] = Field(..., min_length=1)
    dry_run: bool = True

# -------------------------------------------------
# OUTPUT SCHEMA (matches prompt EXACTLY)
# -------------------------------------------------
class StandardizedOutput(BaseModel):
    description: str           # HTML
    short_description: str     # HTML (emoji allowed)
    highlights: str            # HTML
    inclusions: str            # HTML
    additional_info: str       # plain text

# -------------------------------------------------
# LLM CALL (Step 4: timeout/retry/max input length)
# -------------------------------------------------
def call_llm_standardize(
    title: str,
    description_text: str,
    short_description_text: str,
    inclusions_text: str,
    highlights_text: str,
    additional_info_text: str,
) -> StandardizedOutput:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Step 4A: clamp payload sizes
    title = clamp_text(title, 500)
    description_text = clamp_text(description_text, MAX_INPUT_CHARS)
    short_description_text = clamp_text(short_description_text, MAX_INPUT_CHARS)
    inclusions_text = clamp_text(inclusions_text, MAX_INPUT_CHARS)
    highlights_text = clamp_text(highlights_text, MAX_INPUT_CHARS)
    additional_info_text = clamp_text(additional_info_text, MAX_INPUT_CHARS)

    # Step 4B: timeout
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECS)
    master_prompt = load_master_prompt()

    user_payload = {
        "title": title or "",
        "description_raw": description_text or "",
        "short_description_raw": short_description_text or "",
        "inclusions_raw": inclusions_text or "",
        "highlights_raw": highlights_text or "",
        "additional_info_raw": additional_info_text or "",
    }

    system_msg = (
        "You are a content standardization engine for TRVLR travel products. "
        "Follow the master prompt EXACTLY. "
        "Return ONLY valid JSON. No markdown. No explanations."
    )

    # Step 4C: retries (with JSON repair inside first failure)
    for attempt in range(max(1, OPENAI_RETRIES)):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": master_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )

        content = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(content)
            return StandardizedOutput(**data)
        except (json.JSONDecodeError, ValidationError):
            # repair once (on the first attempt only)
            if attempt == 0:
                repair = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "Fix and return ONLY valid JSON. No markdown. No explanations."},
                        {"role": "user", "content": content},
                    ],
                )
                content2 = (repair.choices[0].message.content or "").strip()
                try:
                    data2 = json.loads(content2)
                    return StandardizedOutput(**data2)
                except Exception:
                    pass

            # if still invalid, try next retry (or fail after last)
            if attempt == max(1, OPENAI_RETRIES) - 1:
                raise RuntimeError("LLM output invalid JSON or failed schema validation")

    raise RuntimeError("LLM call failed after retries")

# -------------------------------------------------
# FASTAPI
# -------------------------------------------------
app = FastAPI(title="TRVLR Content Standardizer")

# -------------------------------------------------
# TRVLR API HELPERS
# -------------------------------------------------
def trvlr_headers() -> Dict[str, str]:
    if not TRVLR_API_TOKEN:
        raise RuntimeError("Missing TRVLR_API_TOKEN")

    headers = {
        "Authorization": f"Token {TRVLR_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # IMPORTANT: match browser behaviour
    if TRVLR_ORG_ID:
        headers["organisation"] = TRVLR_ORG_ID

    return headers

def http_get(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers=trvlr_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def http_patch(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.patch(url, headers=trvlr_headers(), json=payload, timeout=60)
    if not r.ok:
        raise HTTPException(
            status_code=r.status_code,
            detail={
                "trvlr_status": r.status_code,
                "trvlr_body": r.text,
                "sent_payload": payload,
            },
        )
    return r.json()

# -------------------------------------------------
# TEXT UTILITIES
# -------------------------------------------------
def strip_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    text = soup.get_text("\n")
    return re.sub(r"[ \t]+", " ", text).strip()
def remove_emojis(s: str) -> str:
    if not s:
        return ""
    # Covers most emoji blocks + variation selectors
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical
        "\U0001F780-\U0001F7FF"  # geometric extended
        "\U0001F800-\U0001F8FF"  # arrows
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess etc
        "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
        "\u2600-\u26FF"          # misc symbols
        "\u2700-\u27BF"          # dingbats
        "\uFE0F"                 # variation selector
        "\u200D"                 # zero-width joiner
        "]+",
        flags=re.UNICODE
    )
    # remove emojis and tidy up spacing
    cleaned = emoji_re.sub("", s)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned
def strip_parentheticals(title: str) -> str:
    # Remove anything like "(includes ...)" completely
    return re.sub(r"\s*\([^)]*\)\s*", " ", title or "").strip()


def fix_common_encoding(title: str) -> str:
    # Common mojibake seen in copied content
    s = title or ""
    s = s.replace("â€“", "–").replace("â€”", "—").replace("Â", "")
    return s


def normalize_whitespace(title: str) -> str:
    s = (title or "").replace("\u00a0", " ")  # NBSP
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_separators(title: str) -> str:
    # Normalize separators to reduce randomness
    s = title or ""
    # make " - " consistent (but avoid breaking times like 15:15)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*:\s*", ": ", s)
    s = normalize_whitespace(s)
    return s


def to_24h_time_token(raw: str) -> str:
    """
    Convert raw time token to HH:MM (24h).
    Handles: 9am, 09:00, 3:15pm, 15:15, 09.00, 3.15 pm
    """
    t = raw.strip().lower().replace(" ", "")

    # 09.00 -> 09:00
    t = t.replace(".", ":")

    # 15:15 stays
    m24 = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", t)
    if m24:
        hh = int(m24.group(1))
        mm = m24.group(2)
        return f"{hh:02d}:{mm}"

    # 9am / 9:00am / 3:15pm
    m = re.fullmatch(r"(\d{1,2})(?::([0-5]\d))?(am|pm)", t)
    if not m:
        return raw  # unknown format; return as-is
    hh = int(m.group(1))
    mm = int(m.group(2) or "0")
    ampm = m.group(3)
    if ampm == "pm" and hh != 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    return f"{hh:02d}:{mm:02d}"


def extract_times(title: str) -> List[str]:
    """
    Return normalized HH:MM tokens found in the title.
    """
    s = title or ""
    tokens: List[str] = []

    # match 9am / 9:00am / 3:15pm / 09:00 / 15:15 / 09.00
    patterns = [
        r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b",
        r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, s, flags=re.I):
            norm = to_24h_time_token(m.group(0))
            if re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", norm) and norm not in tokens:
                tokens.append(norm)

    return tokens


def remove_time_strings(title: str) -> str:
    s = title or ""
    # remove time-like patterns (leave other text intact)
    s = re.sub(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", " ", s, flags=re.I)
    s = re.sub(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b", " ", s)
    return normalize_whitespace(s)


def normalize_title(title: str) -> Dict[str, Any]:
    """
    Deterministically standardize weird titles.
    - Strips parentheticals completely (per your instruction)
    - Extracts variant tokens: session, time, deck, seating, lounge, etc.
    - Produces: Base — token • token • token
    """
    original = title or ""

    s = fix_common_encoding(original)
    s = strip_parentheticals(s)         # REMOVE (...) completely
    s = normalize_separators(s)

    # Extract time tokens first
    times = extract_times(s)
    s_wo_time = remove_time_strings(s)

    # Known tokens (expand as you observe patterns)
    token_specs = [
        ("Morning Cruise", [r"\bmorning cruise\b", r"\bmorning\b"]),
        ("Afternoon Cruise", [r"\bafternoon cruise\b", r"\bafternoon\b"]),
        ("Premier", [r"\bpremier\b"]),
        ("Upper Deck", [r"\bupper deck\b"]),
        ("Main Deck", [r"\bmain deck\b"]),
        ("Window Seating", [r"\bwindow seating\b", r"\bwindow\b"]),
        ("Centre Seating", [r"\bcentre seating\b", r"\bcenter seating\b", r"\bcentre\b", r"\bcenter\b"]),
        ("Central", [r"\bcentral\b"]),
        ("Vista Lounge", [r"\bvista lounge\b"]),
        ("Sky Lounge", [r"\bsky lounge\b"]),
        ("Adults Only", [r"\badults only\b"]),
        ("Bring Your Own Lunch", [r"\bbring your own lunch\b"]),
        ("Cruise Only", [r"\bcruise only\b"]),
        ("Full Day Tour", [r"\bfull day tour\b"]),
        ("Day Tour", [r"\bday tour\b"]),
        ("One Way", [r"\bone way\b"]),
        ("Return", [r"\breturn\b"]),
        ("Shuttle", [r"\bshuttle\b"]),
    ]

    found_tokens: List[str] = []
    s_work = s_wo_time.lower()

    def remove_patterns(text: str, pats: List[str]) -> str:
        out = text
        for p in pats:
            out = re.sub(p, " ", out, flags=re.I)
        return normalize_whitespace(out)

    base_candidate = s_wo_time

    # Heuristic: base is usually before first " - "
    if " - " in base_candidate:
        left, right = base_candidate.split(" - ", 1)
        base_guess = left.strip()
        remainder = right.strip()
    else:
        base_guess = base_candidate.strip()
        remainder = ""

    # Token extraction from both base_guess+remainder (people put tokens everywhere)
    combined = f"{base_candidate} {remainder}".strip()

    for token, patterns in token_specs:
        if any(re.search(p, combined, flags=re.I) for p in patterns):
            if token not in found_tokens:
                found_tokens.append(token)
            combined = remove_patterns(combined, patterns)

    # Choose base title:
    # Prefer the original left side of " - " if it’s not too generic; otherwise use cleaned combined start
    base = base_guess
    base = normalize_whitespace(base)

    # If base is suspiciously short/generic (e.g. "9am Morning Cruise ..."), fall back
    if len(base) < 6 or re.search(r"\b(morning|afternoon|cruise|upper|main|deck|seating)\b", base, flags=re.I):
        # Try using the remaining cleaned text as base
        fallback = normalize_whitespace(combined)
        if fallback:
            base = fallback

    # Build tokens in a consistent order:
    # Session -> Time(s) -> Deck -> Seating -> Others
    ordered: List[str] = []

    def add_if_present(t: str):
        if t in found_tokens and t not in ordered:
            ordered.append(t)

    add_if_present("Morning Cruise")
    add_if_present("Afternoon Cruise")
    add_if_present("Premier")

    # time tokens next (sorted)
    for t in sorted(times):
        if t not in ordered:
            ordered.append(t)

    add_if_present("Upper Deck")
    add_if_present("Main Deck")
    add_if_present("Window Seating")
    add_if_present("Centre Seating")
    add_if_present("Central")

    # lounge and misc
    for t in [
        "Vista Lounge", "Sky Lounge", "Adults Only", "Bring Your Own Lunch",
        "Cruise Only", "Full Day Tour", "Day Tour", "One Way", "Return", "Shuttle"
    ]:
        add_if_present(t)

    # Final title formatting
    if ordered:
        normalized = f"{base} — " + " • ".join(ordered)
    else:
        normalized = base

    normalized = normalize_whitespace(normalized)

    return {
        "original_title": original,
        "normalized_title": normalized,
        "base_title": base,
        "tokens": ordered,
        "times": sorted(times),
    }

# -------------------------------------------------
# DEPARTURE EXTRACTION
# -------------------------------------------------
def extract_and_remove_departures(text: str) -> Tuple[Dict[str, List[str]], str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    out_lines: List[str] = []
    dep: Dict[str, List[str]] = {}
    current_year = None
    in_block = False

    header_re = re.compile(r"\b(departures?|departure dates|operating dates|dates)\b", re.I)
    year_re = re.compile(r"^(20\d{2})\b")
    dateish_re = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9})\b")

    for line in lines:
        y = year_re.search(line)
        if y:
            current_year = y.group(1)
            if header_re.search(line):
                in_block = True
                continue

        if header_re.search(line):
            in_block = True
            continue

        if in_block:
            if not dateish_re.search(line) and not year_re.search(line):
                in_block = False
                out_lines.append(line)
                continue

            bucket = current_year or "Unknown year"
            dep.setdefault(bucket, [])
            cleaned = line.strip("-• ").strip()
            if cleaned and cleaned not in dep[bucket]:
                dep[bucket].append(cleaned)
            continue

        out_lines.append(line)

    return dep, "\n".join(out_lines).strip()

def departures_to_year_bullets(dep_by_year: Dict[str, List[str]]) -> List[str]:
    bullets = []
    for year in sorted(dep_by_year.keys()):
        dates = dep_by_year[year]
        if not dates:
            continue
        compact = "; ".join(dates)
        if year == "Unknown year":
            bullets.append(f"Departures: {compact}")
        else:
            bullets.append(f"{year} departures: {compact}")
    return bullets

# -------------------------------------------------
# SAFETY NET (logs)
# -------------------------------------------------
def log_backup(pk: int, original: Dict[str, Any], patch_payload: Dict[str, Any]) -> str:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"logs/backup_{pk}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"pk": pk, "original": original, "patch_payload": patch_payload},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path

# -------------------------------------------------
# STEP 5: Refactor core logic into reusable function
# -------------------------------------------------
def process_one_pk(pk: int, dry_run: bool) -> Dict[str, Any]:
    url = f"{TRVLR_API_BASE}/entity/travelproduct/{pk}/"
    tp = http_get(url)

    original = {
        "description": tp.get("description", ""),
        "short_description": tp.get("short_description", ""),
        "highlights": tp.get("highlights", ""),
        "inclusions": tp.get("inclusions", ""),
        "additional_info": tp.get("additional_info", ""),
    }

    # Extract departures
    desc_text = strip_html(original["description"])
    dep_by_year, desc_wo_dep = extract_and_remove_departures(desc_text)
    dep_bullets = departures_to_year_bullets(dep_by_year)

    # LLM call
    std = call_llm_standardize(
        title=tp.get("title", ""),
        description_text=desc_wo_dep,
        short_description_text=strip_html(original["short_description"]),
        inclusions_text=strip_html(original["inclusions"]),
        highlights_text=strip_html(original["highlights"]),
        additional_info_text=strip_html(original["additional_info"]),
    )

    # p -> div short_description (your rule)
    short_desc = (std.short_description or "").strip()
    short_desc = remove_emojis(short_desc)
    if short_desc.startswith("<p>") and short_desc.endswith("</p>"):
        short_desc = "<div>" + short_desc[3:-4] + "</div>"

    # Merge departures into additional_info
    dep_block = "\n".join(f"- {b}" for b in dep_bullets) if dep_bullets else ""
    merged_additional = (std.additional_info or "").strip()
    if dep_block:
        merged_additional = (merged_additional + "\n" + dep_block) if merged_additional else dep_block

    # Required fields safety (these were required by TRVLR in your earlier errors)
    organisation_id: Optional[int] = (
        tp.get("organisation", {}).get("id")
        or tp.get("organisation_id")
        or (int(TRVLR_ORG_ID) if TRVLR_ORG_ID else None)
    )
    if organisation_id is None:
        raise RuntimeError("Missing organisation_id (set TRVLR_ORG_ID or ensure tp.organisation.id exists)")

    patch_payload = {
        "pk": pk,
        "title": tp.get("title"),
        "product_type": tp.get("product_type"),
        "organisation_id": organisation_id,

        "description": std.description,
        "short_description": short_desc,
        "highlights": std.highlights,
        "inclusions": std.inclusions,
        "additional_info": merged_additional,
    }

    if dry_run:
        return {"pk": pk, "dry_run": True, "original": original, "patch_payload": patch_payload}

    backup_path = log_backup(pk, original, patch_payload)
    patched = http_patch(url, patch_payload)
    return {"pk": pk, "dry_run": False, "backup_path": backup_path, "patched": patched}

# -------------------------------------------------
# Single endpoint (now reuses process_one_pk)
# -------------------------------------------------
@app.post("/standardize/travelproduct/{pk}")
def standardize(pk: int, dry_run: bool = Query(True)):
    return process_one_pk(pk, dry_run=dry_run)

# -------------------------------------------------
# STEP 6: Batch endpoint (per-item failure safe)
# -------------------------------------------------
@app.post("/standardize/travelproducts")
def standardize_batch(req: BatchRequest):
    if len(req.pks) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many pks. Max is {MAX_BATCH_SIZE}. You sent {len(req.pks)}.",
        )

    results: List[Dict[str, Any]] = []
    for pk in req.pks:
        try:
            out = process_one_pk(pk, dry_run=req.dry_run)
            results.append({"ok": True, **out})
        except Exception as e:
            # per-item failure: do NOT abort whole batch
            results.append({"ok": False, "pk": pk, "error": str(e)})

    return {
        "ok": True,
        "dry_run": req.dry_run,
        "count": len(req.pks),
        "results": results,
    }

class TitleBatchRequest(BaseModel):
    pks: List[int] = Field(..., min_length=1)
    dry_run: bool = True


@app.post("/standardize/travelproduct-title/{pk}")
def standardize_title(pk: int, dry_run: bool = Query(True)):
    url = f"{TRVLR_API_BASE}/entity/travelproduct/{pk}/"
    tp = http_get(url)

    old_title = tp.get("title", "") or ""
    norm = normalize_title(old_title)

    patch_payload = {
        "pk": pk,
        "title": norm["normalized_title"],
        # Keep required fields if TRVLR requires them for PATCH (same as your content patch)
        "product_type": tp.get("product_type"),
        "organisation_id": tp.get("organisation", {}).get("id") or tp.get("organisation_id") or (int(TRVLR_ORG_ID) if TRVLR_ORG_ID else None),
    }

    if dry_run:
        return {
            "dry_run": True,
            "pk": pk,
            "old_title": old_title,
            "new_title": norm["normalized_title"],
            "tokens": norm["tokens"],
            "patch_payload": patch_payload,
        }

    backup_path = log_backup(pk, {"title": old_title}, patch_payload)
    patched = http_patch(url, patch_payload)
    return {
        "dry_run": False,
        "pk": pk,
        "backup_path": backup_path,
        "patched": patched,
    }


@app.post("/standardize/travelproduct-titles")
def standardize_titles_batch(req: TitleBatchRequest):
    # reuse your MAX_BATCH_SIZE if you have it; otherwise default safety
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "50"))
    if len(req.pks) > max_batch:
        raise HTTPException(status_code=400, detail=f"Too many pks. Max {max_batch} per batch.")

    results: List[Dict[str, Any]] = []
    for pk in req.pks:
        try:
            out = standardize_title(pk, dry_run=req.dry_run)  # reuse single logic
            results.append({"ok": True, **out})
        except Exception as e:
            results.append({"ok": False, "pk": pk, "error_type": type(e).__name__, "error": str(e)})

    return {
        "dry_run": req.dry_run,
        "count": len(req.pks),
        "results": results,
        "summary": {
            "succeeded": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
        },
    }
