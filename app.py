import gradio as gr

from langchain_core.messages import BaseMessage

from llm_agent import chat as llm_chat

import pandas as pd


def _display_df(df: pd.DataFrame | None, n: int = 20):
    if df is None or df.empty:
        return pd.DataFrame()
    disp = df.head(n).copy()
    if "synopsis" in disp.columns:
        disp["synopsis"] = disp["synopsis"].fillna("").str.slice(0, 120) + "..."
    return disp


def respond(user_text: str, chat_display, history_state, df_state):
    """Pass user input to LLM agent and update chat, history, and table."""
    assistant_reply, new_history, df = llm_chat(user_text, history_state)

    if df is not None:
        df_state = df

    # Update Gradio chat display
    if chat_display is None:
        chat_display = []

    # Ensure messages format as list of dicts
    chat_display.append({"role": "user", "content": user_text})
    chat_display.append({"role": "assistant", "content": assistant_reply})

    table_df = _display_df(df_state)

    return chat_display, new_history, table_df, df_state


with gr.Blocks(title="Anime Assistant (GPT-4o-mini + Tools)") as demo:
    chatbot = gr.Chatbot(type="messages")

    state_history = gr.State([])  # List[BaseMessage]
    state_df = gr.State(pd.DataFrame())

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Ask me anything about anime...", scale=4)
        send = gr.Button("Send", scale=1)

    send.click(respond, inputs=[txt, chatbot, state_history, state_df], outputs=[chatbot, state_history, table := gr.Dataframe(interactive=False), state_df])


def main():
    demo.launch()


if __name__ == "__main__":
    main()
