#!/usr/bin/env python3
"""
Interactive demo for the Enhanced Anime Recommendation Agent using OpenAI GPT-4o mini

This script provides an enhanced chat interface with proactive tool usage and dataframe management.
The agent will use tools immediately when you mention anime names, genres, or preferences!

Make sure to set your OpenAI API key before running:
  Option 1: Create .env file with: OPENAI_API_KEY=your-key-here
  Option 2: Set environment variable: export OPENAI_API_KEY='your-key-here'
"""

import os
import sys

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

from react_agent import create_agent

def print_welcome():
    """Print welcome message and how the enhanced agent works"""
    print("🎌 Welcome to Your Enhanced Anime Recommendation Assistant!")
    print("=" * 75)
    print("🚀 Hi! I'm your proactive anime recommendation buddy, powered by GPT-4o mini")
    print()
    print("✨ **NEW ENHANCED FEATURES:**")
    print("🧠 INTELLIGENT INPUT DETECTION - I automatically detect when you want recommendations:")
    print("   • 'What do you recommend?' → RECOMMENDATION MODE")
    print("   • 'What should I watch?' → RECOMMENDATION MODE")
    print("   • 'RECOMMEND NOW' → RECOMMENDATION MODE")
    print("   • 'Give me suggestions' → RECOMMENDATION MODE")
    print("   • Regular conversation → CONVERSATION MODE (no recommendations)")
    print()
    print("🔥 DUAL-PROMPT SYSTEM:")
    print("   • CONVERSATION MODE: Uses tools for context, asks questions, builds understanding")
    print("   • RECOMMENDATION MODE: Provides 3-5 specific anime recommendations")
    print("   • The system automatically chooses the right mode based on your input!")
    print()
    print("📊 WORKING DATAFRAME - I maintain a working set of anime that I refine:")
    print("   • Show progress: 'I found 150 action anime, interesting!'")
    print("   • Build on previous searches")
    print("   • Keep track of what we've discovered together")
    print()
    print("💬 **How to interact with me:**")
    print("• CONVERSATION: 'I loved Attack on Titan' → I'll provide context and ask questions")
    print("• CONVERSATION: 'I like action anime' → I'll search and learn about your preferences")
    print("• CONVERSATION: 'I prefer recent anime' → I'll gather info and build understanding")
    print("• RECOMMENDATION: 'What do you recommend?' → I'll provide specific suggestions")
    print("• RECOMMENDATION: 'What should I watch?' → I'll give you anime recommendations")
    print()
    print("🎯 **The Magic:**")
    print("• I automatically detect what you want!")
    print("• No need to specify modes - just talk naturally!")
    print("• I'll have conversations until you ask for recommendations!")
    print()
    print("⚡ **Try these examples:**")
    print("• 'I really enjoyed Death Note' (→ CONVERSATION)")
    print("• 'I want action anime' (→ CONVERSATION)")
    print("• 'What do you recommend?' (→ RECOMMENDATION)")
    print("• 'Show me some options' (→ RECOMMENDATION)")
    print("• 'RECOMMEND NOW' (→ FORCE RECOMMENDATION)")
    print()
    print("📈 **I'll show you my working process:**")
    print("   • What mode I'm in")
    print("   • How many anime I'm considering")
    print("   • What information I've gathered")
    print()
    print("Type 'quit' or 'exit' to end our conversation.")
    print("=" * 75)

def check_api_key():
    """Check if OpenAI API key is available"""
    openai_key = os.getenv('OPENAI_API_KEY')

    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found!")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("\nYou can get an API key from:")
        print("  • OpenAI: https://platform.openai.com/api-keys")
        print("\nNote: GPT-4o mini is perfect for conversations and very cost-effective!")
        return False

    return True

def main():
    """Main interactive demo loop with enhanced state management"""
    print_welcome()

    # Check API key
    if not check_api_key():
        sys.exit(1)

    try:
        # Create enhanced agent with GPT-4o mini
        print("🚀 Initializing your enhanced anime recommendation assistant...")
        agent = create_agent()  # Defaults to gpt-4o-mini
        print("✅ Ready to chat! I'll use tools proactively to help you!\n")

        # Enhanced interactive chat loop with state management
        conversation_state = None
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                if conversation_count == 0:
                    user_input = input("You: ").strip()
                else:
                    user_input = input("\nYou: ").strip()

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q', 'bye', 'goodbye']:
                    print("\n🎌 Thanks for chatting! Hope you find some amazing anime to watch!")
                    print("✨ Happy watching! 📺")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Get enhanced agent response with state management
                print("\n🤖 Assistant: ", end="", flush=True)
                response, conversation_state = agent.chat(user_input, conversation_state)
                print(response)

                # Show mode and working dataframe status
                if conversation_state:
                    mode = "RECOMMENDATION MODE" if conversation_state.get("is_recommendation_request") else "CONVERSATION MODE"
                    print(f"\n🔧 [Mode: {mode}]")
                    
                    if not conversation_state["working_df"].empty:
                        df_size = len(conversation_state["working_df"])
                        search_info = conversation_state["last_search_info"]
                        print(f"📊 [Working with {df_size} anime based on: {search_info}]")
                    else:
                        print("📊 [No working dataframe yet]")
                
                # Show user preferences if any have been tracked
                if conversation_state and conversation_state["user_preferences"]:
                    prefs = conversation_state["user_preferences"]
                    if prefs:
                        print(f"💭 [Preferences tracked: {', '.join(f'{k}: {v}' for k, v in prefs.items())}]")

                conversation_count += 1

            except KeyboardInterrupt:
                print("\n\n🎌 Thanks for chatting! Goodbye! ✨")
                break
            except Exception as e:
                print(f"\n❌ Oops, something went wrong: {e}")
                print("💭 Let's try that again, or type 'quit' to exit.\n")

    except Exception as e:
        print(f"❌ Failed to initialize your enhanced anime assistant: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that your OPENAI_API_KEY is valid and has sufficient credits")
        print("3. Ensure the anime CSV file exists at the correct path")
        print("4. Verify you have a stable internet connection")
        print("5. Try running the basic test: python react_agent.py")

if __name__ == "__main__":
    main() 