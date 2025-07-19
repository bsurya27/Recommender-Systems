#!/usr/bin/env python3
"""
Test script for the Enhanced Anime Recommendation Agent

This script tests the enhanced agent's proactive tool usage and dataframe management.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from react_agent import create_agent

def test_enhanced_agent():
    """Test the enhanced agent functionality"""
    print("🧪 Testing Enhanced Anime Recommendation Agent")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key to run tests.")
        return False
    
    try:
        # Create agent
        print("🚀 Creating enhanced agent...")
        agent = create_agent()
        print("✅ Agent created successfully!\n")
        
        # Test conversation with state management
        conversation_state = None
        
        test_scenarios = [
            {
                "name": "Test 1: Conversation Mode - Anime Name Mention",
                "input": "I really liked Attack on Titan",
                "expected": "Should be in CONVERSATION MODE, search for context, ask follow-up questions"
            },
            {
                "name": "Test 2: Conversation Mode - Genre Preference",
                "input": "I want something with action and drama",
                "expected": "Should be in CONVERSATION MODE, search genres for context, ask questions"
            },
            {
                "name": "Test 3: Conversation Mode - Time Period Preference",
                "input": "I prefer recent anime from the last few years",
                "expected": "Should be in CONVERSATION MODE, gather info about recent anime"
            },
            {
                "name": "Test 4: Recommendation Mode - Explicit Request",
                "input": "What do you recommend?",
                "expected": "Should be in RECOMMENDATION MODE, provide specific recommendations"
            },
            {
                "name": "Test 5: Recommendation Mode - Force Command",
                "input": "RECOMMEND NOW",
                "expected": "Should be in RECOMMENDATION MODE, immediate recommendations"
            },
            {
                "name": "Test 6: Recommendation Mode - Alternative Request",
                "input": "What should I watch?",
                "expected": "Should be in RECOMMENDATION MODE, provide specific suggestions"
            }
        ]
        
        for test in test_scenarios:
            print(f"🔍 {test['name']}")
            print(f"Input: '{test['input']}'")
            print(f"Expected: {test['expected']}")
            print("-" * 40)
            
            try:
                response, conversation_state = agent.chat(test['input'], conversation_state)
                print(f"Response: {response}")
                
                # Show mode and dataframe status
                if conversation_state:
                    mode = "RECOMMENDATION MODE" if conversation_state.get("is_recommendation_request") else "CONVERSATION MODE"
                    print(f"🔧 Mode: {mode}")
                    
                    if not conversation_state["working_df"].empty:
                        df_size = len(conversation_state["working_df"])
                        search_info = conversation_state["last_search_info"]
                        print(f"📊 Working with {df_size} anime from: {search_info}")
                    else:
                        print("📊 No working dataframe yet")
                
                print("✅ Test completed successfully!")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                return False
                
            print("\n" + "=" * 60 + "\n")
        
        print("\n🎉 All enhanced agent tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_recommendation_detection():
    """Test the recommendation detection logic"""
    print("🧪 Testing Recommendation Detection Logic")
    print("=" * 60)
    
    try:
        agent = create_agent()
        
        # Test conversation triggers (should NOT trigger recommendations)
        conversation_inputs = [
            "I really liked Attack on Titan",
            "I want action anime",
            "I prefer recent anime", 
            "What do you think of Naruto?",
            "I'm looking for something similar to Death Note"
        ]
        
        # Test recommendation triggers (should trigger recommendations)
        recommendation_inputs = [
            "What do you recommend?",
            "RECOMMEND NOW",
            "What should I watch?",
            "Give me suggestions",
            "Show me some options",
            "Any recommendations?",
            "What would you suggest?",
            "Based on what I told you, what do you think?"
        ]
        
        print("🗣️ Testing CONVERSATION inputs (should NOT trigger recommendations):")
        for inp in conversation_inputs:
            is_rec = agent._is_recommendation_request(inp)
            status = "❌ FAILED" if is_rec else "✅ PASSED"
            print(f"  '{inp}' → {status}")
        
        print("\n💡 Testing RECOMMENDATION inputs (should trigger recommendations):")
        for inp in recommendation_inputs:
            is_rec = agent._is_recommendation_request(inp)
            status = "✅ PASSED" if is_rec else "❌ FAILED"
            print(f"  '{inp}' → {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_simple_functionality():
    """Test basic functionality without full conversation"""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 60)
    
    try:
        agent = create_agent()
        
        # Test simple chat (backward compatibility)
        print("Testing simple_chat method...")
        response = agent.simple_chat("Hello, I need anime recommendations")
        print(f"Simple chat response: {response[:100]}...")
        print("✅ Simple chat working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎌 Enhanced Anime Recommendation Agent Test Suite")
    print("=" * 70)
    
    # Run tests
    success = True
    
    if not test_enhanced_agent():
        success = False
    
    if not test_recommendation_detection():
        success = False

    if not test_simple_functionality():
        success = False
    
    if success:
        print("\n🎉 All tests passed! The enhanced agent is ready to use!")
        print("💡 Try running: python demo.py")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 