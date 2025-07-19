#!/usr/bin/env python3
"""
Test script for the Gradio UI

This script tests if the Gradio UI can be imported and initialized properly.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

def test_gradio_import():
    """Test if Gradio can be imported"""
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
        return True
    except ImportError:
        print("❌ Gradio not installed")
        return False

def test_agent_import():
    """Test if the agent can be imported"""
    try:
        from react_agent import create_agent
        print("✅ Agent imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False

def test_ui_functions():
    """Test if UI functions can be imported"""
    try:
        from gradio_ui import initialize_agent, create_gradio_interface
        print("✅ UI functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ UI functions import failed: {e}")
        return False

def test_interface_creation():
    """Test if the Gradio interface can be created"""
    try:
        from gradio_ui import create_gradio_interface
        demo = create_gradio_interface()
        print("✅ Gradio interface created successfully")
        return True
    except Exception as e:
        print(f"❌ Interface creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Gradio UI Components")
    print("=" * 40)
    
    tests = [
        ("Gradio Import", test_gradio_import),
        ("Agent Import", test_agent_import),
        ("UI Functions", test_ui_functions),
        ("Interface Creation", test_interface_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gradio UI is ready to use!")
        print("💡 Run 'python gradio_ui.py' to launch the interface")
    else:
        print("❌ Some tests failed. Please check the setup.")
        if not os.getenv('OPENAI_API_KEY'):
            print("💡 Note: Some tests may fail without OPENAI_API_KEY set")

if __name__ == "__main__":
    main() 