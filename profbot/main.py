import sys
import uuid

from langgraph.types import Command

from chat import run_chat
from graph import build_graph


def run(name, college, refresh=False):
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Looking up {name} ({college})...")
    app.invoke({"name": name, "college": college, "refresh": refresh}, config)

    # the graph pauses when it needs a human decision; keep resuming until it's done
    while True:
        snap = app.get_state(config)
        if not snap.next:
            break
        payload = snap.tasks[0].interrupts[0].value
        print(f"\n{payload['message']}")
        for o in payload["options"]:
            print(f"  [{o['i']}] {o['name']} | {o['works']} works | {o['institutions']} | {o['topics']}")
        choice = input("Pick a number: ").strip()
        app.invoke(Command(resume=choice), config)

    final = app.get_state(config).values
    if final.get("error"):
        raise SystemExit(final["error"])
    if final.get("result", {}).get("filter_dropped"):
        print("(Few papers matched the field filter, so the filter was dropped for this profile.)")

    run_chat(final["profile"])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--refresh"]
    refresh = "--refresh" in sys.argv
    if len(args) >= 2:
        run(args[0], args[1], refresh)
    else:
        run(input("Professor name: ").strip(), input("College: ").strip(), refresh)