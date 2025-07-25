import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Tuple, Any
import re
import pandas as pd
from langgraph.graph import StateGraph, END

# Import master tool list only
from tools import tools_list

########################################################
# Build a registry mapping tool names to tool objects
########################################################

TOOL_REGISTRY = {t.name: t for t in tools_list}

# ------------------ Helper parsing ------------------

GENRES = [
    "action", "adventure", "comedy", "drama", "fantasy", "horror", "mystery",
    "romance", "sci-fi", "slice of life", "sports", "supernatural", "thriller",
]


def parse_user(user_message: str) -> Tuple[str | None, Dict[str, Any]]:
    """Infer which tool to call and its arguments from `user_message`."""
    msg = user_message.lower()

    # Similarity search: phrases like "similar to <title>" or "like <title>"
    m_sim = re.search(r"(?:similar to|like) ([\w\s:,'-]+)", msg)
    if m_sim:
        return "recommend_similar_anime", {"anime_name": m_sim.group(1).strip(), "top_n": 10}

    # Genre route
    for g in GENRES:
        if g in msg:
            return "get_anime_ids_by_genre", {"genre": g}

    # Year after / before
    if m := re.search(r"after (\d{4})", msg):
        return "get_anime_ids_after_year", {"year": int(m.group(1))}
    if m := re.search(r"before (\d{4})", msg):
        return "get_anime_ids_before_year", {"year": int(m.group(1))}

    # Synopsis keyword (fallback to first 5+ char word)
    words = re.findall(r"[a-zA-Z]{5,}", msg)
    if words:
        return "search_anime_ids_by_synopsis", {"keyword": words[0]}

    return None, {}


# ------------------ LangGraph Nodes ------------------

State = Dict[str, Any]  # keys: user_message, df, anime_ids, tool_name, tool_args


def router_node(state: State) -> State:
    tool_name, args = parse_user(state["user_message"])
    state["tool_name"] = tool_name
    state["tool_args"] = args
    return state


def tool_exec_node(state: State) -> State:
    name = state.get("tool_name")
    if not name:
        # No tool to run
        state["anime_ids"] = []
        return state

    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        print(f"⚠️  Unknown tool requested: {name}")
        state["anime_ids"] = []
        return state

    print(f"🔧 Executing tool: {name}")
    result = tool_fn.invoke(state.get("tool_args", {}))

    # Some tools return DataFrames (e.g., get_anime_details)
    if isinstance(result, pd.DataFrame):
        state["df"] = result
        state["anime_ids"] = result["anime_id"].tolist() if "anime_id" in result.columns else []
    else:
        state["anime_ids"] = result  # assume list[int] or similar
    return state


def details_node(state: State) -> State:
    """Ensure we have a DataFrame with complete details for the collected IDs."""
    if "df" in state and isinstance(state["df"], pd.DataFrame) and not state["df"].empty:
        return state

    ids = state.get("anime_ids", [])
    if not ids:
        state["df"] = pd.DataFrame()
        return state

    details_tool = TOOL_REGISTRY.get("get_anime_details")
    if details_tool is None:
        print("⚠️  get_anime_details tool missing from registry")
        state["df"] = pd.DataFrame()
        return state

    state["df"] = details_tool.invoke({"anime_ids": ids})
    return state


# ------------------ Graph assembly ------------------

sg = StateGraph(State)
sg.add_node("router", router_node)
sg.add_node("run_tool", tool_exec_node)
sg.add_node("details", details_node)

sg.set_entry_point("router")
sg.add_edge("router", "run_tool")
sg.add_edge("run_tool", "details")
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
    return out.get("df", pd.DataFrame()), out.get("anime_ids", [])


if __name__ == "__main__":
    while True:
        try:
            um = input("Query: ")
        except (KeyboardInterrupt, EOFError):
            break
        df, ids = run_agent(um)
        print("Found", len(ids), "anime. Sample ids:", ids[:10]) 