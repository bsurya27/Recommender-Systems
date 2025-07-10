#!/usr/bin/env python3
"""
Quick test of the RECOMMEND NOW keyword feature

This demonstrates how the force command works to get immediate recommendations.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_recommend_now():
    """Test the RECOMMEND NOW force command"""
    
    print("⚡ Testing 'RECOMMEND NOW' Force Command")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        print("📝 Example of how it would work:")
        print()
        print("You: Hi! I loved Attack on Titan")
        print("Agent: Great choice! What kind of shows do you usually enjoy?")
        print("You: RECOMMEND NOW")
        print("Agent: Based on your love for Attack on Titan, here are 5 anime...")
        print("       [immediately uses tools to find recommendations]")
        return
    
    try:
        from react_agent import create_agent
        agent = create_agent()
        
        print("🧪 Running quick test conversation...")
        print()
        
        # Simulate a conversation with the force command
        messages = [
            "Hi! I loved Attack on Titan and Death Note",
            "RECOMMEND NOW"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"👤 You: {message}")
            response = agent.chat(message)
            print(f"🤖 Agent: {response}")
            print()
            
        print("✅ Test completed! The 'RECOMMEND NOW' command should have")
        print("   triggered immediate recommendations using your preferences.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run: pip install -r requirements.txt")

def show_keyword_info():
    """Show information about the keyword feature"""
    print("💡 About 'RECOMMEND NOW'")
    print("=" * 30)
    print("🔥 FORCE COMMAND: Makes the agent recommend immediately")
    print("⚡ Works anytime: Even with minimal info")
    print("🎯 Case insensitive: 'recommend now', 'RECOMMEND NOW', etc.")
    print("🚀 Bypasses questions: No more chat, straight to recommendations")
    print()
    print("📋 Other ways to get recommendations:")
    print("• 'What do you recommend?'")
    print("• 'Show me some options'")
    print("• 'Give me suggestions'")
    print("• Natural conversation (agent offers automatically)")
    print()
    print("🎌 Perfect for when you're ready and don't want more questions!")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Test 'RECOMMEND NOW' with API")
    print("2. Learn about the keyword feature")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        test_recommend_now()
    elif choice == "2":
        show_keyword_info()
    else:
        print("Invalid choice. Showing keyword info:")
        show_keyword_info() 