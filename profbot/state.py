from typing import TypedDict

class ProfState(TypedDict, total=False):
    name: str
    college: str
    refresh: bool
    candidates: list     # all institution matches, best first
    options: list        # the serious ones, shown if we have to ask
    ambiguous: bool
    chosen_id: str
    result: dict         # raw search output
    profile: dict        # distilled profile
    error: str



    
