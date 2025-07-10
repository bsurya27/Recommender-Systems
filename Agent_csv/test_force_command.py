#!/usr/bin/env python3
"""
Test script specifically for the RECOMMEND NOW force command

This verifies that the agent actually uses tools and makes recommendations
when the force command is used.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_force_command_with_preferences():
    """Test RECOMMEND NOW with some preferences shared"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - cannot test")
        return False
    
    try:
        from react_agent import create_agent
        agent = create_agent()
        
        print("🧪 Testing RECOMMEND NOW with preferences...")
        print("=" * 50)
        
        # First, share some preferences
        print("👤 You: I loved Attack on Titan and Death Note")
        response1 = agent.chat("I loved Attack on Titan and Death Note")
        print(f"🤖 Agent: {response1}")
        print()
        
        # Now use the force command
        print("👤 You: RECOMMEND NOW")
        response2 = agent.chat("RECOMMEND NOW")
        print(f"🤖 Agent: {response2}")
        
        # Check if tools were actually used
        if "I'll" in response2 or "Let me" in response2 or "Here are" in response2:
            print("\n✅ SUCCESS: Agent appears to be making recommendations!")
            return True
        else:
            print("\n❌ FAILED: Agent is still asking questions instead of recommending")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_force_command_minimal_info():
    """Test RECOMMEND NOW with minimal information"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - cannot test")
        return False
    
    try:
        from react_agent import create_agent
        agent = create_agent()
        
        print("\n🧪 Testing RECOMMEND NOW with minimal info...")
        print("=" * 50)
        
        # Use force command immediately
        print("👤 You: RECOMMEND NOW")
        response = agent.chat("RECOMMEND NOW")
        print(f"🤖 Agent: {response}")
        
        # Check if tools were actually used
        if "I'll" in response or "Let me" in response or "Here are" in response:
            print("\n✅ SUCCESS: Agent made recommendations with minimal info!")
            return True
        else:
            print("\n❌ FAILED: Agent asked for more info instead of recommending")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_expected_behavior():
    """Show what the agent should do"""
    print("📋 Expected Behavior for 'RECOMMEND NOW'")
    print("=" * 45)
    print("✅ Should IMMEDIATELY use tools")
    print("✅ Should provide 3-5 specific anime recommendations")
    print("✅ Should include anime details (year, rating, genre)")
    print("✅ Should explain why each anime matches preferences")
    print("❌ Should NOT ask for more information")
    print("❌ Should NOT say 'tell me more about your preferences'")
    print()
    print("🔧 If the agent is still asking questions after 'RECOMMEND NOW',")
    print("   then the force command isn't working properly.")

if __name__ == "__main__":
    print("⚡ RECOMMEND NOW Force Command Test")
    print("This will verify the force command actually works!\n")
    
    show_expected_behavior()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Cannot run live tests without OPENAI_API_KEY")
        print("💡 Set your API key and run this test to verify the fix!")
        exit(1)
    
    print("\n🚀 Running live tests...")
    
    # Test with preferences
    test1_passed = test_force_command_with_preferences()
    
    # Test with minimal info
    test2_passed = test_force_command_minimal_info()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! The force command is working!")
    else:
        print("❌ SOME TESTS FAILED. The force command needs more work.")
        print("💡 The agent should use tools and make recommendations, not ask questions.")
    print("=" * 50) 