import gradio as gr
import pandas as pd
from Agent import run_agent

_df_full = pd.read_csv("Data/anime_clean.csv")


def _display_df(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return a copy of df (head n) with shortened synopsis for nicer table."""
    if df is None or df.empty:
        return pd.DataFrame()
    disp = df.head(n).copy()
    if "synopsis" in disp.columns:
        disp["synopsis"] = disp["synopsis"].fillna("").str.slice(0, 120) + "..."
    return disp


def _pairs_from_messages(msgs):
    pairs = []
    temp_user = None
    for m in msgs:
        if m["role"] == "user":
            temp_user = m["content"]
        elif m["role"] == "assistant" and temp_user is not None:
            pairs.append((temp_user, m["content"]))
            temp_user = None
    return pairs


def respond(message, messages, df_state):
    """Handle user message via agent and update UI state."""
    if messages is None:
        messages = []
    # Append user message
    messages.append({"role": "user", "content": message})

    # Use LangGraph agent to update DataFrame
    new_df, _ids = run_agent(message, df_state)
    if _ids:
        assistant_reply = (
            "Great! I've found some shows that fit. "
            "Any other details—like timeframe, tone, or length—that you'd prefer?"
        )
    else:
        assistant_reply = (
            "Interesting! Could you tell me more, e.g., a genre you enjoy, "
            "a specific year range, or themes you're in the mood for?"
        )

    messages.append({"role": "assistant", "content": assistant_reply})
    return messages, _display_df(new_df), messages, new_df


def recommend(messages, df_state):
    """Create a recommendation list from the current DataFrame."""
    if df_state is None or df_state.empty:
        response = "I don't have enough information yet to recommend titles. Tell me more about your tastes!"
        messages.append({"role": "assistant", "content": response})
        return messages, _display_df(df_state), messages, df_state

    rec_df = df_state.sort_values("rating", ascending=False).head(10).reset_index(drop=True)
    titles = rec_df["name"].tolist()
    response = "Here are some anime you might enjoy:\n" + "\n".join(titles)
    messages.append({"role": "assistant", "content": response})
    return messages, _display_df(rec_df, n=10), messages, df_state


with gr.Blocks(title="Anime Recommender") as demo:
    chatbot = gr.Chatbot(type="messages")

    state_history = gr.State([])  # list of message dicts
    state_df = gr.State(_df_full)

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Tell me what kind of anime you like...", scale=4)
        send = gr.Button("Send", scale=1)
        rec_btn = gr.Button("Recommend", scale=1)

    send.click(respond, inputs=[txt, state_history, state_df], outputs=[chatbot, table := gr.Dataframe(interactive=False), state_history, state_df])
    rec_btn.click(recommend, inputs=[state_history, state_df], outputs=[chatbot, table, state_history, state_df])

    # Initial table value
    table.value = _display_df(_df_full)


def main():
    demo.launch()


if __name__ == "__main__":
    main()
