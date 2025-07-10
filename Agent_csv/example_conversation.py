#!/usr/bin/env python3
"""
Example conversation with the Anime Recommendation Assistant

This script demonstrates how the agent naturally converses to understand
preferences and provide personalized recommendations.

Note: This requires OPENAI_API_KEY to be set to actually run.
"""

from react_agent import create_agent

def example_conversation():
    """Simulate a conversation with the recommendation agent"""
    
    print("🎌 Example Conversation with Anime Recommendation Assistant")
    print("=" * 65)
    print("This demonstrates how the agent learns your preferences and makes recommendations.\n")
    
    # Example conversation messages
    conversation = [
        {
            "user": "Hi! I'm looking for some anime recommendations",
            "expected_response": "The agent should greet you warmly and ask about your preferences"
        },
        {
            "user": "I really loved Attack on Titan and Death Note. Both were so intense!",
            "expected_response": "The agent should recognize you like psychological thrillers and ask follow-up questions"
        },
        {
            "user": "I want something recent, maybe from the last few years, and with great animation",
            "expected_response": "The agent should ask more questions or start offering to make recommendations"
        },
        {
            "user": "What do you recommend based on what I told you?",
            "expected_response": "The agent should use tools to find recent anime matching your preferences and give 3-5 specific recommendations with explanations"
        },
        {
            "user": "Those sound good! Do you have anything more on the action side?",
            "expected_response": "The agent should search for action anime and provide more specific recommendations"
        }
    ]
    
    print("💭 Sample conversation flow:")
    print("(This shows what a real conversation would look like)\n")
    
    for i, turn in enumerate(conversation, 1):
        print(f"👤 User: {turn['user']}")
        print(f"🤖 Expected: {turn['expected_response']}")
        print()
    
    print("🎯 Ways to trigger recommendations:")
    print("• '**RECOMMEND NOW**' - FORCE COMMAND (immediate recommendations!)")
    print("• 'What do you recommend?'")
    print("• 'Show me some options'") 
    print("• 'What should I watch?'")
    print("• 'Give me suggestions'")
    print("• 'Based on what I told you, what do you think?'")
    print("• Or just chat - the agent will offer recommendations naturally!")
    print()
    print("💡 Pro tip: 'RECOMMEND NOW' works anytime to force immediate suggestions!")
    print()
    print("🔧 To see this in action:")
    print("1. Set your OPENAI_API_KEY")
    print("2. Run: python demo.py")
    print("3. Start chatting with the agent!")
    print("\n✨ The agent will learn your preferences and find perfect anime for you!")

def test_conversation_with_api():
    """Test actual conversation if API key is available"""
    import os
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - showing example only")
        example_conversation()
        return
    
    print("🚀 Running actual conversation test...")
    try:
        agent = create_agent()
        
        test_messages = [
            "Hi! I need anime recommendations",
            "I loved Attack on Titan - so intense and well-animated!",
            "I want something recent with great action scenes"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {message}")
            response = agent.chat(message)
            print(f"🤖 Assistant: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your API key is valid and you have internet connection")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Show example conversation (no API key needed)")
    print("2. Test with actual API (requires OPENAI_API_KEY)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        example_conversation()
    elif choice == "2":
        test_conversation_with_api()
    else:
        print("Invalid choice. Showing example conversation:")
        example_conversation() 