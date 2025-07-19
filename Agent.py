import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Tuple, Any
import re
import pandas as pd
from langgraph.graph import StateGraph, END

# Import tool functions
from tools import (
    get_anime_ids_before_year,
    get_anime_ids_after_year,
    get_anime_ids_by_genre,
    search_anime_ids_by_synopsis,
    get_anime_details,
)

# ------------------ Helper parsing ------------------
GENRES = [
    "action", "adventure", "comedy", "drama", "fantasy", "horror", "mystery",
    "romance", "sci-fi", "slice of life", "sports", "supernatural", "thriller",
]

def parse_user(user_message: str) -> Tuple[str, Dict[str, Any]]:
    """Return route name and arguments dict based on heuristics."""
    msg = user_message.lower()
    # Genre route
    for g in GENRES:
        if g in msg:
            return "genre", {"genre": g}
    # Year after
    m_after = re.search(r"after (\d{4})", msg)
    if m_after:
        return "year_after", {"year": int(m_after.group(1))}
    # Year before
    m_before = re.search(r"before (\d{4})", msg)
    if m_before:
        return "year_before", {"year": int(m_before.group(1))}
    # Synopsis keyword route (take first keyword)
    words = re.findall(r"[a-zA-Z]{4,}", msg)
    if words:
        return "synopsis", {"keyword": words[0]}
    return "none", {}

# ------------------ LangGraph Nodes ------------------

State = Dict[str, Any]  # keys: user_message, df, anime_ids, route, tool_args


def router(state: State) -> State:
    route, args = parse_user(state["user_message"])
    state["route"] = route
    state["tool_args"] = args
    return state


def genre_tool(state: State) -> State:
    args = state["tool_args"]
    ids = get_anime_ids_by_genre.invoke(args)
    state["anime_ids"] = ids
    return state


def after_tool(state: State) -> State:
    ids = get_anime_ids_after_year.invoke(state["tool_args"])
    state["anime_ids"] = ids
    return state


def before_tool(state: State) -> State:
    ids = get_anime_ids_before_year.invoke(state["tool_args"])
    state["anime_ids"] = ids
    return state


def synopsis_tool(state: State) -> State:
    ids = search_anime_ids_by_synopsis.invoke(state["tool_args"])
    state["anime_ids"] = ids
    return state


def details_node(state: State) -> State:
    ids: List[int] = state.get("anime_ids", [])
    if ids:
        state["df"] = get_anime_details.invoke({"anime_ids": ids})
    else:
        state["df"] = pd.DataFrame()
    return state

# Build graph
sg = StateGraph(State)
sg.add_node("router", router)
sg.add_node("genre", genre_tool)
sg.add_node("after", after_tool)
sg.add_node("before", before_tool)
sg.add_node("synopsis", synopsis_tool)
sg.add_node("details", details_node)

sg.set_entry_point("router")

sg.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        "genre": "genre",
        "year_after": "after",
        "year_before": "before",
        "synopsis": "synopsis",
        "none": "details",
    },
)

# Each tool node -> details -> END
for node in ["genre", "after", "before", "synopsis"]:
    sg.add_edge(node, "details")
sg.add_edge("details", END)

workflow = sg.compile()

# ----------------- Public function ------------------

def run_agent(user_message: str, df_state: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, List[int]]:
    """Run the graph and return updated DataFrame and list of IDs."""
    if df_state is None:
        df_state = pd.DataFrame()
    state_in: State = {
        "user_message": user_message,
        "df": df_state,
        "anime_ids": [],
    }
    out = workflow.invoke(state_in)
    return out["df"], out.get("anime_ids", [])

if __name__ == "__main__":
    while True:
        try:
            um = input("Query: ")
        except (KeyboardInterrupt, EOFError):
            break
        df, ids = run_agent(um)
        print("Found", len(ids), "anime. Sample ids:", ids[:10]) 