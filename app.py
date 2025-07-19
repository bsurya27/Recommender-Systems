import gradio as gr
import pandas as pd
from prompt import system_prompt

_df = pd.read_csv("Data/anime_clean.csv")

def respond(message, history, df_state):
    history = history + [(message, None)]
    lower = message.lower()
    genres = []
    for g in [
        "action","drama","romance","comedy","adventure","fantasy","sci-fi","slice of life","horror","mystery","supernatural","sports","shounen","seinen","shoujo","josei"
    ]:
        if g in lower:
            genres.append(g)
    if genres:
        mask = df_state["genre"].fillna("").str.contains("|".join(genres), case=False, na=False)
        df_state = df_state[mask]
    reply = "Got it. Tell me more about what you like."
    history[-1] = (message, reply)
    return history, df_state, df_state

def recommend(history, df_state):
    rec_df = df_state.sort_values("rating", ascending=False).head(10).reset_index(drop=True)
    titles = rec_df["name"].tolist()
    response = "Here are some anime you may enjoy:\n" + "\n".join(titles)
    history = history + [("", response)]
    return history, rec_df, df_state

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    table = gr.Dataframe(value=_df.head(20), interactive=False)
    text = gr.Textbox()
    df_state = gr.State(_df)
    text.submit(respond, inputs=[text, chatbot, df_state], outputs=[chatbot, table, df_state])
    rec_btn = gr.Button("Recommend")
    rec_btn.click(recommend, inputs=[chatbot, df_state], outputs=[chatbot, table, df_state])

def main():
    demo.launch()
