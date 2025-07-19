#!/usr/bin/env python3
"""
Test script to verify your .env file and OpenAI API key setup

Run this to check if everything is configured correctly before using the agent.
"""

import os
from dotenv import load_dotenv

def test_env_setup():
    """Test if .env file and API key are properly configured"""
    
    print("🔧 Testing .env file setup...")
    print("=" * 50)
    
    # Load .env file
    print("📁 Loading .env file...")
    load_dotenv()
    
    # Check if .env file exists
    env_file_exists = os.path.exists('.env')
    print(f"📄 .env file exists: {'✅ Yes' if env_file_exists else '❌ No'}")
    
    # Check if API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        # Mask the key for security (show first/last few chars)
        if len(api_key) > 10:
            masked_key = api_key[:8] + "..." + api_key[-4:]
        else:
            masked_key = "***"
        
        print(f"🔑 API key loaded: ✅ Yes ({masked_key})")
        print(f"🔍 Key format: {'✅ Correct' if api_key.startswith('sk-') else '❌ Incorrect (should start with sk-)'}")
        
        # Test API connection
        print("\n🌐 Testing API connection...")
        try:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # Simple test message
            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content="Hello! Just testing the connection. Reply with 'OK'.")])
            
            if response and response.content:
                print("✅ API connection successful!")
                print(f"📝 Test response: {response.content}")
                return True
            else:
                print("❌ API responded but with empty content")
                return False
                
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            print("💡 Check your API key and internet connection")
            return False
    
    else:
        print("🔑 API key loaded: ❌ No")
        print("\n💡 To fix this:")
        print("1. Make sure you have a .env file in this directory")
        print("2. Add this line to your .env file:")
        print("   OPENAI_API_KEY=your-actual-api-key-here")
        print("3. Get your API key from: https://platform.openai.com/api-keys")
        return False

def show_env_file_example():
    """Show example .env file content"""
    print("\n📝 Example .env file content:")
    print("=" * 40)
    print("# Your OpenAI API Key")
    print("OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("") 
    print("# Get your key from: https://platform.openai.com/api-keys")
    print("# DO NOT share this file or commit it to git!")

if __name__ == "__main__":
    print("🎌 Anime Recommendation Agent - Environment Test")
    print("This will check if your .env file and API key are set up correctly.\n")
    
    try:
        success = test_env_setup()
        
        if success:
            print("\n🎉 Great! Your setup is working perfectly!")
            print("🚀 You can now run: python demo.py")
        else:
            print("\n❌ Setup needs attention. See messages above.")
            show_env_file_example()
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        show_env_file_example() 