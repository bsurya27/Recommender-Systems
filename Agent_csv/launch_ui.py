#!/usr/bin/env python3
"""
Simple launcher for the Gradio UI

This script provides a quick way to launch the anime recommendation agent's web interface.
"""

import subprocess
import sys
import os

def main():
    """Launch the Gradio UI"""
    print("🎌 Anime Recommendation Agent - UI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("gradio_ui.py"):
        print("❌ Error: gradio_ui.py not found!")
        print("Please run this script from the Agent_csv directory")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import gradio
        print("✅ Gradio is available")
    except ImportError:
        print("❌ Gradio not installed. Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            sys.exit(1)
    
    # Launch the UI
    print("🚀 Launching Gradio UI...")
    try:
        from gradio_ui import main as ui_main
        ui_main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 