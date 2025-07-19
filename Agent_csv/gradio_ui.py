#!/usr/bin/env python3
"""
Gradio UI for the Enhanced Anime Recommendation Agent

This provides a web-based chat interface for the anime recommendation agent
with conversation state management and working dataframe display.
"""

import os
import sys
import gradio as gr
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from react_agent import create_agent

# Global agent instance
agent = None
conversation_state = None

def initialize_agent():
    """Initialize the agent with error handling"""
    global agent
    try:
        if not os.getenv('OPENAI_API_KEY'):
            return False, "❌ Error: OPENAI_API_KEY not set! Please set your OpenAI API key."
        
        agent = create_agent()
        return True, "✅ Agent initialized successfully!"
    except Exception as e:
        return False, f"❌ Error initializing agent: {e}"

def chat_with_agent(message: str, history: list, num_rows: int = 10):
    """
    Chat with the agent and update conversation history
    
    Args:
        message (str): User's message
        history (list): Chat history in OpenAI format
        num_rows (int): Number of rows to show in table
        
    Returns:
        tuple: (history, "", info_display, df_display, table_info)
    """
    global agent, conversation_state
    
    if not agent:
        success, msg = initialize_agent()
        if not success:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": msg})
            df, info = get_sample_dataframe(num_rows)
            return history, "", "", df, info
    
    try:
        # Get response from agent
        response, conversation_state = agent.chat(message, conversation_state)
        
        # Add to history in OpenAI format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Generate info display
        info_display = generate_info_display()
        
        # Get updated dataframe
        df, table_info = get_sample_dataframe(num_rows)
        
        return history, "", info_display, df, table_info
        
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        df, info = get_sample_dataframe(num_rows)
        return history, "", "", df, info

def generate_info_display():
    """Generate information display about current state"""
    global conversation_state
    
    if not conversation_state:
        return "💬 No conversation state yet"
    
    info_parts = []
    
    # Mode detection
    mode = "RECOMMENDATION MODE" if conversation_state.get("is_recommendation_request") else "CONVERSATION MODE"
    info_parts.append(f"🔧 **Mode:** {mode}")
    
    # Tool usage indicator
    info_parts.append(f"⚡ **Tool Usage:** Forced after every interaction")
    
    # Working dataframe info
    if not conversation_state["working_df"].empty:
        df_size = len(conversation_state["working_df"])
        search_info = conversation_state["last_search_info"]
        info_parts.append(f"📊 **Working Dataset:** {df_size} anime")
        if search_info:
            info_parts.append(f"📝 **Last Search:** {search_info}")
    else:
        info_parts.append("📊 **Working Dataset:** No data loaded yet")
    
    # User preferences
    if conversation_state.get("user_preferences"):
        prefs = conversation_state["user_preferences"]
        if prefs:
            info_parts.append("💫 **Preferences Learned:**")
            for key, value in prefs.items():
                info_parts.append(f"   • {key}: {value}")
    
    # Agent status
    info_parts.append("🎯 **Agent Status:** Proactive tool usage enabled")
    
    return "\n".join(info_parts)

def clear_conversation():
    """Clear the conversation history and state"""
    global conversation_state
    conversation_state = None
    return [], ""

def get_recommendations(history):
    """Force the agent to provide recommendations based on current conversation"""
    global agent, conversation_state
    
    if not agent:
        success, msg = initialize_agent()
        if not success:
            history.append({"role": "assistant", "content": msg})
            return history, ""
    
    try:
        # Force recommendation mode with a clear request
        response, conversation_state = agent.chat("What do you recommend based on our conversation?", conversation_state)
        
        # Add to history in OpenAI format
        history.append({"role": "user", "content": "🎯 [Recommendation Request]"})
        history.append({"role": "assistant", "content": response})
        
        # Generate info display
        info_display = generate_info_display()
        
        return history, info_display
        
    except Exception as e:
        error_msg = f"❌ Error getting recommendations: {e}"
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

def get_sample_dataframe(num_rows=10):
    """Get a sample of the working dataframe for display"""
    global conversation_state
    
    if not conversation_state or conversation_state["working_df"].empty:
        return pd.DataFrame({"Message": ["No data loaded yet. Start chatting to see anime data!"]}), "No data available"
    
    df = conversation_state["working_df"]
    
    # Define priority columns for display
    priority_cols = ['name', 'genre', 'type', 'episodes', 'rating', 'scored_by', 'synopsis']
    display_cols = []
    
    # Add priority columns if they exist
    for col in priority_cols:
        if col in df.columns:
            display_cols.append(col)
    
    # Add any remaining columns
    for col in df.columns:
        if col not in display_cols:
            display_cols.append(col)
    
    # Limit to first 8 columns for better display
    display_cols = display_cols[:8]
    
    if display_cols:
        sample_df = df[display_cols].head(num_rows)
        
        # Clean up the dataframe for better display
        sample_df = sample_df.copy()
        
        # Truncate long text fields
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                sample_df[col] = sample_df[col].astype(str).apply(
                    lambda x: x[:100] + "..." if len(str(x)) > 100 else x
                )
        
        # Create info text
        total_rows = len(df)
        info_text = f"📊 Showing {len(sample_df)} of {total_rows} anime entries"
        
        return sample_df, info_text
    else:
        return df.head(num_rows), f"📊 Showing {min(num_rows, len(df))} of {len(df)} entries"

def refresh_dataframe(num_rows=10):
    """Refresh the dataframe display with specified number of rows"""
    df, info = get_sample_dataframe(num_rows)
    return df, info

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="🎌 Anime Recommendation Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎌 Enhanced Anime Recommendation Agent
        
        ### Powered by GPT-4o mini with **Proactive Tool Usage**
        
        **Features:**
        - 🧠 **Intelligent Mode Detection**: Automatically switches between conversation and recommendation modes
        - 🔥 **Dual-Prompt System**: Context-aware responses based on your input
        - ⚡ **Proactive Tool Usage**: **NEW!** Agent automatically uses tools after every interaction
        - 📊 **Interactive Working Dataset**: Sortable, searchable table showing anime the agent is considering
        - 💬 **Natural Conversation**: Ask questions, mention preferences, get recommendations
        - 🎯 **One-Click Recommendations**: Dedicated button for instant suggestions
        
        **How to use:**
        - **Conversation Mode**: "I loved Attack on Titan" or "I like action anime"
        - **Recommendation Mode**: "What do you recommend?" or click "Get Recommendations" button
        - **Quick Recommendations**: Use the dedicated button anytime for instant suggestions
        - **Auto-Tool Usage**: The agent will automatically search for data whenever you mention anime terms!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Start chatting! Tell me about anime you like or ask for recommendations...",
                    label="Chat with Agent",
                    type="messages"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    recommend_btn = gr.Button("Get Recommendations", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
                    
            with gr.Column(scale=1):
                # Info panel
                info_display = gr.Markdown(
                    value="💬 Chat with the agent to see information here",
                    label="Agent Status"
                )
                
                # Interactive dataframe display
                with gr.Group():
                    gr.Markdown("### 📊 Working Dataset")
                    
                    # Controls for the table
                    with gr.Row():
                        show_count = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=5,
                            label="Show rows",
                            scale=2
                        )
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                    
                    # Interactive table
                    df_display = gr.DataFrame(
                        value=pd.DataFrame({"Message": ["Start chatting to see anime data here"]}),
                        label="",
                        interactive=True,
                        wrap=True,
                        datatype=["str"] * 10  # Will be updated dynamically
                    )
                    
                    # Table info
                    table_info = gr.Markdown(
                        value="💡 This table shows anime the agent is currently considering for you",
                        visible=True
                    )
        
        # Event handlers
        def handle_submit(message, history, num_rows):
            if message.strip():
                return chat_with_agent(message, history, num_rows)
            df, info = get_sample_dataframe(num_rows)
            return history, message, "", df, info
        
        def handle_clear():
            clear_conversation()
            return [], "", pd.DataFrame({"Message": ["Chat cleared - start chatting to see anime data!"]}), "No data available"
        
        def handle_recommend(history):
            updated_history, info_display = get_recommendations(history)
            return updated_history, info_display
        
        def handle_refresh(num_rows):
            df, info = refresh_dataframe(num_rows)
            return df, info
        
        def handle_slider_change(num_rows):
            df, info = refresh_dataframe(num_rows)
            return df, info
        
        # Connect events
        msg.submit(handle_submit, [msg, chatbot, show_count], [chatbot, msg, info_display, df_display, table_info])
        submit_btn.click(handle_submit, [msg, chatbot, show_count], [chatbot, msg, info_display, df_display, table_info])
        
        # Update recommendation button handler
        def handle_recommend_with_table(history, num_rows):
            updated_history, info_display = get_recommendations(history)
            df, table_info = get_sample_dataframe(num_rows)
            return updated_history, info_display, df, table_info
        
        recommend_btn.click(handle_recommend_with_table, [chatbot, show_count], [chatbot, info_display, df_display, table_info])
        clear_btn.click(handle_clear, [], [chatbot, info_display, df_display, table_info])
        refresh_btn.click(handle_refresh, [show_count], [df_display, table_info])
        
        # Update dataframe when slider changes
        show_count.change(handle_slider_change, [show_count], [df_display, table_info])
        
        gr.Markdown("""
        ---
        **Tips:**
        - ⚡ **NEW!** The agent now automatically uses tools after every interaction with anime-related terms
        - The agent will search for data whenever you mention anime names, genres, or preferences
        - Ask "What do you recommend?" to get specific suggestions
        - Use the "Get Recommendations" button to force immediate recommendations
        - Use "RECOMMEND NOW" in chat to force immediate recommendations
        
        **Interactive Table Features:**
        - 📊 **Working Dataset**: Shows anime the agent is currently considering for you
        - 🔢 **Row Control**: Use the slider to show 5-100 rows
        - 🔄 **Auto-refresh**: Table updates automatically as you chat
        - 📋 **Sortable**: Click column headers to sort data
        - 🔍 **Searchable**: Type in cells to filter content
        - 📝 **Detailed View**: Hover over truncated text to see full content
        """)
    
    return demo

def main():
    """Main function to run the Gradio interface"""
    print("🎌 Starting Enhanced Anime Recommendation Agent UI")
    print("=" * 60)
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key:")
        print("  Option 1: Create .env file with: OPENAI_API_KEY=your-key-here")
        print("  Option 2: Set environment variable: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    print("🚀 Launching Gradio interface...")
    print("💡 The interface will open in your browser automatically")
    print("🔗 You can also access it at: http://localhost:7860")
    print("📝 Press Ctrl+C to stop the server")
    
    demo.launch(
        share=False,  # Set to True if you want to share publicly
        server_port=7860,
        server_name="0.0.0.0",
        inbrowser=True
    )

if __name__ == "__main__":
    main() 