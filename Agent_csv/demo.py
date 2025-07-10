#!/usr/bin/env python3
"""
Interactive demo for the Anime Recommendation Agent using OpenAI GPT-4o mini

This script provides a simple chat interface to get personalized anime recommendations.
The agent will chat with you to understand your preferences, then recommend anime you'll love!

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
    """Print welcome message and how the agent works"""
    print("🎌 Welcome to Your Personal Anime Recommendation Assistant!")
    print("=" * 75)
    print("🤖 Hi! I'm your anime recommendation buddy, powered by GPT-4o mini")
    print()
    print("💬 How I work:")
    print("• I'll chat with you to understand your anime preferences")
    print("• Tell me about anime you've loved, genres you enjoy, or your mood")
    print("• I'll use my database to find perfect recommendations just for you!")
    print("• We can have a natural conversation - no need for specific commands")
    print()
    print("✨ Just start chatting! Try saying:")
    print("• 'Hi! I need some anime recommendations'")
    print("• 'I loved Attack on Titan, what should I watch next?'")
    print("• 'I want something romantic and recent'")
    print("• 'Show me some good action anime from the last few years'")
    print("• 'I'm new to anime, what should I start with?'")
    print()
    print("🎯 When you're ready for recommendations, say:")
    print("• 'What do you recommend?' or 'Show me some options'")
    print("• 'What should I watch?' or 'Give me suggestions'")
    print("• 'Based on what I told you, what do you think?'")
    print("• Or I'll offer recommendations after we chat a bit!")
    print()
    print("⚡ FORCE RECOMMENDATIONS: Type 'RECOMMEND NOW' to get")
    print("   immediate suggestions based on everything you've told me!")
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
    """Main interactive demo loop"""
    print_welcome()

    # Check API key
    if not check_api_key():
        sys.exit(1)

    try:
        # Create agent with GPT-4o mini
        print("🚀 Initializing your anime recommendation assistant...")
        agent = create_agent()  # Defaults to gpt-4o-mini
        print("✅ Ready to chat! Let's find you some amazing anime!\n")

        # Interactive chat loop
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

                # Get agent response
                print("\n🤖 Assistant: ", end="", flush=True)
                response = agent.chat(user_input)
                print(response)

                conversation_count += 1

            except KeyboardInterrupt:
                print("\n\n🎌 Thanks for chatting! Goodbye! ✨")
                break
            except Exception as e:
                print(f"\n❌ Oops, something went wrong: {e}")
                print("💭 Let's try that again, or type 'quit' to exit.\n")

    except Exception as e:
        print(f"❌ Failed to initialize your anime assistant: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that your OPENAI_API_KEY is valid and has sufficient credits")
        print("3. Ensure the anime CSV file exists at the correct path")
        print("4. Verify you have a stable internet connection")

if __name__ == "__main__":
    main() 