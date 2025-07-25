from __future__ import annotations

"""LangGraph-powered anime assistant that uses GPT-4o-mini and the project tools.

This version avoids LangChain's high-level agent helpers and instead builds a
simple LangGraph workflow with two nodes:

1. llm_node   – calls the LLM (with tool bindings) to plan/respond.
2. tool_node  – executes any tool calls found in the LLM's response.

The graph cycles until the LLM responds without tool calls.
"""

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Sequence, TypedDict, Optional, List, Any

import pandas as pd
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import tools_list

# -----------------------------------------------------------------------------
# LLM configuration
# -----------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools_list)

# -----------------------------------------------------------------------------
# State definition – just the message history
# -----------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    df: Optional[pd.DataFrame]  # latest DataFrame from tools (e.g., get_anime_details)

# -----------------------------------------------------------------------------
# Graph nodes
# -----------------------------------------------------------------------------

def llm_node(state: ChatState):
    """Run the LLM on the conversation so far."""
    response = _llm.invoke(list(state["messages"]))
    return {"messages": [response]}


def _extract_anime_ids(result: Any) -> List[int] | None:
    """Extract anime IDs from various tool result formats."""
    if isinstance(result, list):
        if all(isinstance(x, int) for x in result):
            return result  # Already a list of IDs
        if all(isinstance(x, dict) and 'anime_id' in x for x in result):
            return [x['anime_id'] for x in result]  # List of anime dicts
    return None

def tool_node(state: ChatState) -> dict:
    """Execute tool calls emitted by the LLM."""
    new_messages: list[BaseMessage] = []
    df: Optional[pd.DataFrame] = state.get("df")
    last_msg = state["messages"][-1]
    print("Tool node - processing tool calls...")
    for call in getattr(last_msg, "tool_calls", []) or []:
        tool_name = call.get("name")
        tool_args = call.get("args", {})
        tool_call_id = call.get("id")  # Use the id from the tool call
        print(f"Executing tool: {tool_name} with args: {tool_args}")
        tool = next((t for t in tools_list if t.name == tool_name), None)
        if tool is None:
            new_messages.append(
                ToolMessage(name=tool_name or "unknown", content="ERROR: tool not found", tool_call_id=tool_call_id)
            )
            continue
        try:
            result = tool.invoke(tool_args)
            print(f"Tool result type: {type(result)}")
            
            # Handle different result types
            anime_ids = _extract_anime_ids(result)
            if anime_ids:
                print(f"Extracted {len(anime_ids)} anime IDs: {anime_ids[:5]}...")
                from tools import get_anime_details
                df = get_anime_details.invoke({"anime_ids": anime_ids})
                print(f"Fetched details DataFrame shape: {df.shape}")
            elif isinstance(result, pd.DataFrame):
                print(f"Got DataFrame directly with shape: {result.shape}")
                df = result
        except Exception as e:
            print(f"Tool error: {e}")
            result = f"ERROR running tool: {e}"
        new_messages.append(ToolMessage(name=tool_name, content=str(result), tool_call_id=tool_call_id))
    return {"messages": new_messages, "df": df}

# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

graph = StateGraph(ChatState)

graph.add_node("llm", llm_node)

graph.add_node("tools", tool_node)

graph.set_entry_point("llm")

# If LLM produced tool calls → run tools; else END

def _should_run_tools(state: ChatState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("llm", _should_run_tools)

graph.add_edge("tools", "llm")

_chat_workflow = graph.compile()

# -----------------------------------------------------------------------------
# Public chat helper
# -----------------------------------------------------------------------------

def chat(user_text: str, history: Sequence[BaseMessage] | None = None) -> tuple[str, list[BaseMessage], pd.DataFrame | None]:
    """Send a user message to the assistant and get a reply.

    Parameters
    ----------
    user_text: str
        The latest user message.
    history: list[BaseMessage] | None
        Prior conversation history (LangChain messages).

    Returns
    -------
    tuple[str, list[BaseMessage]]
        assistant reply text and updated history list.
    """
    if history is None:
        history = []

    # Kick off the graph with existing history + new HumanMessage
    state_in: ChatState = {"messages": history + [HumanMessage(content=user_text)], "df": None}
    print("\nStarting chat workflow...")
    result = _chat_workflow.invoke(state_in)
    print("Chat workflow finished")

    # The assistant's final reply is the last AI message
    final_messages = result["messages"]
    assistant_reply = next((m.content for m in reversed(final_messages) if isinstance(m, AIMessage)), "")
    final_df = result.get("df")
    print(f"Final DataFrame: {'shape=' + str(final_df.shape) if final_df is not None else 'None'}")

    return assistant_reply, list(final_messages), final_df 