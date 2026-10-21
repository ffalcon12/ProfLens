import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from search import normalize_title, search_professor

load_dotenv()

MAX_AREAS = 5
CACHE = Path("cache")


# ---------- model backend (swappable) ----------

def make_llm():
    """LLM_BACKEND=claude in .env uses the Claude API; anything else uses local Ollama."""
    if os.getenv("LLM_BACKEND") == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            temperature=0,
            max_tokens=1500,
        )
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0)


llm = make_llm()


# ---------- schemas (one per small task) ----------

class Areas(BaseModel):
    research_areas: list[str] = Field(
        description="3 to 5 specific research areas inferred from the titles, 2-6 words each."
    )


class Assignment(BaseModel):
    title: str = Field(description="A paper title copied exactly from the list.")
    area: str = Field(description="One area copied exactly from the area list, or 'other'.")


class Assignments(BaseModel):
    items: list[Assignment]


class Text(BaseModel):
    text: str = Field(description="Plain-language text, no lists.")


# ---------- helpers ----------

def titles_block(papers):
    return "\n".join(f"- {w['title']}" for w in papers)


def unique_papers(result):
    """top_cited + recent, deduped by title."""
    seen, out = set(), []
    for w in result["top_cited"] + result["recent"]:
        k = normalize_title(w["title"])
        if k not in seen:
            seen.add(k)
            out.append(w)
    return out


# ---------- the small LLM tasks ----------

def get_areas(papers):
    out = llm.with_structured_output(Areas).invoke([
        ("system", "You read paper titles and name the research areas they cover. "
                   "Use only the titles. Be specific, never a broad field like 'Computer Science'."),
        ("user", f"TITLES:\n{titles_block(papers)}"),
    ])
    return out.research_areas[:MAX_AREAS]


def classify(papers, areas):
    """Assign every paper to exactly one area (or 'other'). Titles are verified in code."""
    area_list = "\n".join(f"- {a}" for a in areas)
    out = llm.with_structured_output(Assignments).invoke([
        ("system", "For each paper title, pick the ONE area that fits best. Copy the area "
                   "name exactly from the list. Use 'other' only if no area fits."),
        ("user", f"AREAS:\n{area_list}\n\nTITLES:\n{titles_block(papers)}"),
    ])

    by_key = {normalize_title(w["title"]): w for w in papers}
    valid = set(areas)
    assigned = {}
    for it in out.items:
        k = normalize_title(it.title)
        if k in by_key:                                  # only real titles count
            assigned[k] = it.area if it.area in valid else "other"
    for k in by_key:                                     # anything the model skipped
        assigned.setdefault(k, "other")
    return by_key, assigned


def describe(area, papers):
    out = llm.with_structured_output(Text).invoke([
        ("system", "In ONE sentence, say what these paper titles have in common. "
                   "Use only the titles. Do not claim results or importance."),
        ("user", f"AREA: {area}\nTITLES:\n{titles_block(papers)}"),
    ])
    return out.text


def summarize(name, themes):
    lines = "\n".join(f"- {t['name']} ({len(t['papers'])} papers): {t['description']}" for t in themes)
    out = llm.with_structured_output(Text).invoke([
        ("system", "Write exactly 3 plain sentences for a student summarizing a researcher's focus, using only "
"the themes given. Lead with the theme that has the most papers. No praise words, no filler."),
        ("user", f"RESEARCHER: {name}\nTHEMES:\n{lines}"),
    ])
    return out.text


# ---------- distill ----------

def distill(result, college=None):
    papers = unique_papers(result)

    areas = get_areas(papers)
    by_key, assigned = classify(papers, areas)

    themes = []
    for a in areas:
        members = [by_key[k] for k, v in assigned.items() if v == a]
        if len(members) >= 2:
            themes.append({
                "name": a,
                "description": describe(a, members),
                "papers": [w["title"] for w in members],
            })


    themes.sort(key=lambda t: len(t["papers"]), reverse=True)
    other = [by_key[k]["title"] for k, v in assigned.items() if v == "other"]

    # trend info is counted in code, never written by the model
    def mix(plist):
        c = Counter(assigned[normalize_title(w["title"])] for w in plist)
        return dict(c.most_common())

    area_mix = {"most_cited": mix(result["top_cited"]), "recent": mix(result["recent"])}
    top_recent = [a for a in area_mix["recent"] if a != "other"][:2]
    recent_directions = (
        f"Most of the recent papers fall under: {', '.join(top_recent)}."
        if top_recent else "No clear pattern in the recent papers."
    )

    insts = result["institutions"]
    institution = next(
        (i for i in insts if college and college.lower() in i.lower()),
        insts[0] if insts else "unknown",
    )
    return {
        "name": result["name"],
        "institution": institution,
        "research_areas": areas,
        "themes": themes,
        "other_papers": other,
        "area_mix": area_mix,
        "recent_directions": recent_directions,
        "summary": summarize(result["name"], themes) if themes else "not found",
        "most_cited": result["top_cited"],
        "recent": result["recent"],
    }


# ---------- cache ----------

def _cache_path(name, college):
    slug = f"{name}_{college}".lower().replace(" ", "_")
    return CACHE / f"{slug}.json"


def save_profile(profile, name, college):
    CACHE.mkdir(exist_ok=True)
    path = _cache_path(name, college)
    path.write_text(json.dumps(profile, indent=2, default=str))
    return path


def get_profile(name, college, refresh=False):
    """Load the cached profile, or search + distill + save it. None if no match."""
    path = _cache_path(name, college)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    result = search_professor(name, college, n=10)
    if result is None:
        return None
    profile = distill(result, college)
    save_profile(profile, name, college)
    return profile

def load_cached(name, college):
    path = _cache_path(name, college)
    return json.loads(path.read_text()) if path.exists() else None

# ---------- run directly for experiments ----------

if __name__ == "__main__":
    profile = get_profile("Dongjie Wang", "Kansas", refresh=True)
    if profile is None:
        raise SystemExit("No match.")

    shown = {k: v for k, v in profile.items() if k not in ("most_cited", "recent")}
    print(json.dumps(shown, indent=2))