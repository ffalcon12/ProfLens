import re
import httpx

BASE = "https://api.openalex.org"
EMAIL = "your_email_here"   # puts you in OpenAlex's faster request pool


# ---------- helpers ----------

def _field_id(topic):
    """'https://openalex.org/fields/17' -> '17' (or '' if missing)."""
    return ((topic.get("field") or {}).get("id") or "").split("/")[-1]


def field_share(author, field_id):
    """Fraction of the author's topic activity that falls in the given field."""
    topics = author.get("topics") or []
    total = sum(t["count"] for t in topics)
    if not total:
        return 0.0
    hit = sum(t["count"] for t in topics if _field_id(t) == str(field_id))
    return hit / total


def top_fields(author, min_share=0.10):
    """Fields that each account for at least `min_share` of the author's topic activity."""
    totals, names = {}, {}
    for t in author.get("topics") or []:
        fid = _field_id(t)
        if not fid:
            continue
        totals[fid] = totals.get(fid, 0) + t["count"]
        names[fid] = (t.get("field") or {}).get("display_name", fid)

    grand = sum(totals.values())
    if not grand:
        return []

    return [
        (fid, names[fid])
        for fid, c in sorted(totals.items(), key=lambda x: -x[1])
        if c / grand >= min_share
    ]


def normalize_title(title):
    """Lowercase and strip punctuation so preprint/published copies match."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def get_works(author_id, sort, field_ids=None, n=5):
    """
    Fetch works for one author, deduped by title.
    field_ids: list of field id strings (OR'd together), or None for no filter.
    """
    flt = f"author.id:{author_id}"
    if field_ids:
        flt += ",primary_topic.field.id:" + "|".join(field_ids)

    r = httpx.get(
        f"{BASE}/works",
        params={
            "filter": flt,
            "sort": sort,
            "per-page": 25,   # fetch extra so dedupe still leaves n
            "mailto": EMAIL,
            "select": "id,title,publication_year,cited_by_count,doi",
        },
        timeout=30,
    )
    r.raise_for_status()

    seen, out = set(), []
    for w in r.json()["results"]:
        key = normalize_title(w.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(w)
        if len(out) == n:
            break
    return out


# ---------- main function ----------

def find_candidates(name, college, prefer_field=17):
    """All candidates whose institution matches the college, best first."""
    r = httpx.get(
        f"{BASE}/authors",
        params={
            "search": name,
            "mailto": EMAIL,
            "per-page": 50,
            "select": "id,display_name,works_count,cited_by_count,last_known_institutions,topics",
        },
        timeout=30,
    )
    r.raise_for_status()

    matches = []
    for a in r.json()["results"]:
        insts = a.get("last_known_institutions") or []
        if any(college.lower() in i["display_name"].lower() for i in insts):
            matches.append(a)

    def score(a):
        share = field_share(a, prefer_field) if prefer_field is not None else 0.0
        return (share, a["works_count"])

    matches.sort(key=score, reverse=True)
    return matches


def serious_candidates(matches, prefer_field=17):
    """Candidates big enough and on-field enough to be the real person."""
    if not matches:
        return []
    best = matches[0]
    best_share = field_share(best, prefer_field) if prefer_field is not None else 0.0
    min_works = max(5, 0.2 * best["works_count"])

    return [
        a for a in matches
        if a["works_count"] >= min_works
        and (field_share(a, prefer_field) if prefer_field is not None else 0.0) >= 0.5 * best_share
    ]


def build_result(author, n_matches=1, min_share=0.10, n=5, use_filter=True):
    """Fetch works for one chosen author and return the result dict."""
    author_id = author["id"].split("/")[-1]
    fields = top_fields(author, min_share) if use_filter else []
    field_ids = [fid for fid, _ in fields]

    return {
        "name": author["display_name"],
        "id": author["id"],
        "works_count": author["works_count"],
        "cited_by_count": author["cited_by_count"],
        "institutions": [i["display_name"] for i in (author.get("last_known_institutions") or [])],
        "topics": [t["display_name"] for t in (author.get("topics") or [])],
        "fields_used": [nm for _, nm in fields],
        "n_matches": n_matches,
        "top_cited": get_works(author_id, "cited_by_count:desc", field_ids or None, n),
        "recent": get_works(author_id, "publication_date:desc", field_ids or None, n),
    }


def search_professor(name, college, prefer_field=17, min_share=0.10, n=5):
    """Old one-call interface, kept so distill.py and the experiments still work."""
    matches = find_candidates(name, college, prefer_field)
    if not matches:
        return None
    return build_result(matches[0], len(matches), min_share, n)

# ---------- run directly for experiments ----------

def show(result):
    if result is None:
        print("No match.")
        return
    print(result["name"], result["id"], f"({result['n_matches']} matches)")
    print("Institutions:", result["institutions"])
    print("Topics:", result["topics"][:3])
    print("Fields used for works filter:", result["fields_used"])

    print("\nMost cited:")
    for w in result["top_cited"]:
        print(" ", w["publication_year"], w["cited_by_count"], w["title"])

    print("\nMost recent:")
    for w in result["recent"]:
        print(" ", w["publication_year"], w["cited_by_count"], w["title"])


if __name__ == "__main__":
    tests = [
        ("Hongyang Sun", "Kansas"),
        ("Tyrone Duncan", "Kansas"),
        ("Dongjie Wang", "Kansas"),
    ]
    for name, college in tests:
        print("=" * 70)
        show(search_professor(name, college))