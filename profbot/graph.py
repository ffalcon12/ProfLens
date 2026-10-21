from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from distill import distill, load_cached, save_profile
from search import build_result, find_candidates, serious_candidates
from state import ProfState


# ---------- nodes ----------

def check_cache(state):
    if not state.get("refresh"):
        cached = load_cached(state["name"], state["college"])
        if cached:
            return {"profile": cached}
    return {}


def find(state):
    matches = find_candidates(state["name"], state["college"])
    if not matches:
        return {"error": f"No match for '{state['name']}' at '{state['college']}'."}
    options = serious_candidates(matches)
    return {
        "candidates": matches,
        "options": options,
        "ambiguous": len(options) > 1,
        "chosen_id": matches[0]["id"],
    }


def pick(state):
    """Pause the graph and ask the human which profile is the right person."""
    shown = [
        {
            "i": i,
            "name": a["display_name"],
            "works": a["works_count"],
            "institutions": [x["display_name"] for x in (a.get("last_known_institutions") or [])],
            "topics": [t["display_name"] for t in (a.get("topics") or [])[:3]],
        }
        for i, a in enumerate(state["options"])
    ]
    choice = interrupt({"message": "Several profiles could be this person. Pick one.", "options": shown})
    return {"chosen_id": state["options"][int(choice)]["id"]}


def fetch(state):
    author = next(a for a in state["candidates"] if a["id"] == state["chosen_id"])
    n_matches = len(state["candidates"])

    result = build_result(author, n_matches, n=10)
    # thin profile: the field filter may be hiding real work, so retry without it
    if len(result["top_cited"]) < 3:
        result = build_result(author, n_matches, n=10, use_filter=False)
        result["filter_dropped"] = True
    return {"result": result}


def distill_node(state):
    profile = distill(state["result"], state["college"])
    save_profile(profile, state["name"], state["college"])
    return {"profile": profile}


# ---------- routing ----------

def after_cache(state):
    return END if state.get("profile") else "find"


def after_find(state):
    if state.get("error"):
        return END
    return "pick" if state.get("ambiguous") else "fetch"


# ---------- graph ----------

def build_graph():
    g = StateGraph(ProfState)
    g.add_node("check_cache", check_cache)
    g.add_node("find", find)
    g.add_node("pick", pick)
    g.add_node("fetch", fetch)
    g.add_node("distill", distill_node)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges("check_cache", after_cache, ["find", END])
    g.add_conditional_edges("find", after_find, ["pick", "fetch", END])
    g.add_edge("pick", "fetch")
    g.add_edge("fetch", "distill")
    g.add_edge("distill", END)

    return g.compile(checkpointer=MemorySaver())