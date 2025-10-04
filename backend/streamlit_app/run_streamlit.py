#!/usr/bin/env python3
"""
Streamlit app runner with authentication integration
"""
import os
import sys
import subprocess
from pathlib import Path

# Enable development mode for iframe access
os.environ['STREAMLIT_DEV_MODE'] = 'true'

def main():
    # Set environment variables for Django integration
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medisense.settings')
    
    # Get Django secret key from settings
    try:
        import django
        from django.conf import settings
        django.setup()
        os.environ['DJANGO_SECRET_KEY'] = settings.SECRET_KEY
    except Exception as e:
        print(f"Warning: Could not load Django settings: {e}")
        print("Make sure to set DJANGO_SECRET_KEY environment variable")
    
    # Run Streamlit app
    app_path = Path(__file__).parent / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    print("Starting Streamlit app...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
