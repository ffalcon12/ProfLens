import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from distill import get_profile, llm
from search import normalize_title

ABOUT_ME = Path("about_me.txt")   # your own blurb; kept out of git
MAX_HISTORY = 10                  # messages kept in the chat window


# ---------- profile -> context text ----------

def profile_context(p):
    lines = [
        f"Professor: {p['name']} ({p['institution']})",
        f"Research areas: {'; '.join(p['research_areas'])}",
        "",
        "Themes:",
    ]
    for t in p["themes"]:
        lines.append(f"- {t['name']}: {t['description']}")
        for title in t["papers"]:
            lines.append(f"    * {title}")
    lines += ["", f"Summary: {p['summary']}", f"Recent directions: {p['recent_directions']}", ""]

    lines.append("Most cited papers (year | citations | title):")
    for w in p["most_cited"]:
        lines.append(f"- {w['publication_year']} | {w['cited_by_count']} | {w['title']}")
    lines.append("Most recent papers (year | citations | title):")
    for w in p["recent"]:
        lines.append(f"- {w['publication_year']} | {w['cited_by_count']} | {w['title']}")

    if p.get("other_papers"):
        lines.append("Papers that did not fit the themes (may be unrelated or misattributed):")
        lines += [f"- {t}" for t in p["other_papers"]]
    return "\n".join(lines)


CHAT_SYSTEM = """You answer questions about one professor's research, using ONLY the profile below.
Rules:
- If the answer is not in the profile, say you don't have that information. Do not guess.
- The profile has paper titles, years and citation counts only, no abstracts. Do not claim
  to know a paper's methods or results; say what its title suggests.
- You have no information about courses taught, lab openings, or whether they take students.
- When you mention a paper, copy its title exactly as written in the profile.
- Keep answers short and plain.

PROFILE:
{context}"""


EMAIL_SYSTEM = """You draft a short cold email from a student to a professor.
Rules:
- Under 150 words. Plain and polite. No flattery, no filler, no "eager to contribute".
- Every claim about the student must come from the STUDENT BLURB. Never state or imply
  an interest, skill or experience that is not written there.
- If CANDIDATE PAPERS is not empty, pick exactly ONE of them, use its exact title, say what
  the title says it is about, and connect it to something in the STUDENT BLURB.
- If CANDIDATE PAPERS is empty, write a shorter, plainer email with no paper mention.
- Sign off with the student's full name from the blurb. Never write bracketed placeholders.
- End with one specific ask based on the student's goal.
- Output the subject line and the email body only.

PROFILE:
{context}

STUDENT BLURB:
{about}"""


# ---------- LLM helper ----------

def ask(messages):
    """Stream the model's reply to the terminal and return the full text."""
    parts = []
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        parts.append(chunk.content)
    print()
    return "".join(parts)


# ---------- commands that need no LLM ----------

def show_papers(title, papers):
    print(f"\n{title}")
    for w in papers:
        print(f"  {w['publication_year']} | {w['cited_by_count']:>4} cites | {w['title']}")
        if w.get("doi"):
            print(f"      {w['doi']}")
    print()


def titles_in_text(profile, text):
    """Return profile paper titles that appear in the text."""
    all_titles = [w["title"] for w in profile["most_cited"] + profile["recent"]]
    norm_text = normalize_title(text)
    return [t for t in all_titles if normalize_title(t) in norm_text]


# ---------- email ----------

def draft_email(profile, goal):
    if not ABOUT_ME.exists():
        print("Create about_me.txt first: background, your work, interests, what you're looking for.\n")
        return
    about = ABOUT_ME.read_text().strip()

    print()
    reply = ask([
        SystemMessage(content=EMAIL_SYSTEM.format(context=profile_context(profile), about=about)),
        HumanMessage(content=f"My goal: {goal}"),
    ])

    found = titles_in_text(profile, reply)
    if found:
        print(f"\n(Verified paper title(s) in this draft: {len(found)}. Read the paper before sending.)\n")
    else:
        print("\n(No exact paper title found in this draft. Check any paper reference by hand.)\n")


HELP = """Commands:
  /papers        most cited papers
  /recent        most recent papers
  /email <goal>  draft an email, e.g. /email ask about a research position
  /help          show this
  /quit          exit
Anything else is a question about the professor's research."""


# ---------- chat loop ----------

def run_chat(profile):
    """Chat loop over an already-built profile. main.py calls this after the graph finishes."""
    print(f"\n{profile['name']}, {profile['institution']}\n{profile['summary']}\n")
    print(HELP + "\n")

    system = SystemMessage(content=CHAT_SYSTEM.format(context=profile_context(profile)))
    history = []

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user in ("/quit", "/exit"):
            break
        if user == "/help":
            print(HELP + "\n")
        elif user == "/papers":
            show_papers("Most cited:", profile["most_cited"])
        elif user == "/recent":
            show_papers("Most recent:", profile["recent"])
        elif user.startswith("/email"):
            goal = user[len("/email"):].strip() or "ask about research opportunities"
            draft_email(profile, goal)
        else:
            history.append(HumanMessage(content=user))
            print()
            reply = ask([system] + history[-MAX_HISTORY:])
            print()
            history.append(AIMessage(content=reply))


def main(name, college):
    """Standalone entry point that skips the graph (kept for quick tests)."""
    print(f"Loading profile for {name} ({college})...")
    profile = get_profile(name, college)
    if profile is None:
        raise SystemExit("No match found.")
    run_chat(profile)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(input("Professor name: ").strip(), input("College: ").strip())