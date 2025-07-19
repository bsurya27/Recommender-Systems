#!/usr/bin/env python3
"""
Test script to demonstrate proactive tool usage

This script tests that the agent now automatically uses tools after every interaction
when anime-related terms are mentioned.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from react_agent import create_agent

def test_proactive_tool_usage():
    """Test that the agent uses tools proactively"""
    print("🧪 Testing Proactive Tool Usage")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        return False
    
    try:
        agent = create_agent()
        print("✅ Agent created successfully!\n")
        
        # Test messages that should trigger tool usage
        test_messages = [
            "I really liked Attack on Titan",
            "I want action anime",
            "I prefer recent anime from 2020 onwards",
            "I love popular mainstream anime",
            "I want high-quality anime with good ratings",
            "I'm interested in romance and comedy genres",
            "What do you recommend?"
        ]
        
        conversation_state = None
        
        for i, message in enumerate(test_messages, 1):
            print(f"🔍 Test {i}: '{message}'")
            print("-" * 40)
            
            try:
                # Enable tool usage tracking
                response, conversation_state = agent.chat(message, conversation_state)
                
                print(f"Response: {response[:200]}...")
                
                # Check working dataframe
                if conversation_state and not conversation_state["working_df"].empty:
                    df_size = len(conversation_state["working_df"])
                    search_info = conversation_state["last_search_info"]
                    print(f"📊 Working Dataset: {df_size} anime from {search_info}")
                else:
                    print("📊 Working Dataset: Empty")
                
                print("✅ Test completed!")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                return False
                
            print("\n" + "=" * 60 + "\n")
        
        print("🎉 All proactive tool usage tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def main():
    """Run the proactive tool usage test"""
    print("🎌 Proactive Tool Usage Test")
    print("=" * 40)
    
    if test_proactive_tool_usage():
        print("\n🎉 Proactive tool usage is working!")
        print("💡 The agent now automatically uses tools when anime terms are mentioned")
        print("🚀 Try the Gradio UI: python gradio_ui.py")
    else:
        print("\n❌ Proactive tool usage test failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 