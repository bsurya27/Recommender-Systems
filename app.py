import gradio as gr
import pandas as pd
from langchain_core.messages import BaseMessage
from llm_agent import chat as llm_chat


def _display_df(df: pd.DataFrame | None, n: int = 20):
    if df is None or df.empty:
        return pd.DataFrame(columns=["anime_id", "name", "genre", "rating"])
    disp = df.head(n).copy()
    if "synopsis" in disp.columns:
        disp["synopsis"] = disp["synopsis"].fillna("").str.slice(0, 120) + "..."
    return disp


def respond(user_text: str, chat_display, history_state, df_state):
    """Pass user input to LLM agent and update chat, history, and table."""
    print("Before chat - df shape:", df_state.shape if df_state is not None else "None")
    
    # Get response and potential new DataFrame from agent
    assistant_reply, new_history, new_df = llm_chat(user_text, history_state)
    print("After chat - new_df shape:", new_df.shape if new_df is not None else "None")

    # Update chat display
    if chat_display is None:
        chat_display = []
    chat_display.append({"role": "user", "content": user_text})
    chat_display.append({"role": "assistant", "content": assistant_reply})

    # Update DataFrame state and table view
    if new_df is not None and not new_df.empty:
        df_state = new_df
        print("Updating table with new data")
    
    table_df = _display_df(df_state)
    print("Table display shape:", table_df.shape)

    return chat_display, new_history, table_df, df_state


with gr.Blocks(title="Anime Assistant (GPT-4o-mini + Tools)") as demo:
    # Chat interface
    chatbot = gr.Chatbot(type="messages")
    
    # State containers
    state_history = gr.State([])  # List[BaseMessage]
    state_df = gr.State(pd.DataFrame(columns=["anime_id", "name", "genre", "rating"]))
    
    # Input row
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Ask me anything about anime...",
            scale=4
        )
        send = gr.Button("Send", scale=1)
    
    # Results table
    table = gr.Dataframe(
        value=pd.DataFrame(columns=["anime_id", "name", "genre", "rating"]),
        interactive=False,
        visible=True,
        wrap=True,
    )
    
    # Wire up the send button
    send.click(
        respond,
        inputs=[txt, chatbot, state_history, state_df],
        outputs=[chatbot, state_history, table, state_df],
        show_progress=True,
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
