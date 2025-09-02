#!/usr/bin/env python3
"""
ATS Resume Analyzer - Backend Startup Script

This script helps users easily start the enhanced backend server with proper
environment setup and dependency checking.

Features:
- Automatic dependency installation
- Environment variable validation
- Service health checks
- User-friendly error messages and setup guidance

Usage:
    python start_backend.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("    ATS Resume Analyzer - Enhanced Backend Server")
    print("    Version 2.0.0 with Gemini AI Integration")
    print("="*60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")
    return True


def check_requirements():
    """Check if requirements.txt exists and install dependencies"""
    print("\n📦 Checking dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found")
        print("   Please ensure you're running this script from the project directory.")
        return False
    
    try:
        print("📥 Installing/updating dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False


def check_environment():
    """Check environment variables and configuration"""
    print("\n🔧 Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env file not found, but .env.example exists")
            print("   Please copy .env.example to .env and configure your settings:")
            print("   cp .env.example .env")
            print("\n   Required configurations:")
            print("   - GEMINI_API_KEY: Get from https://makersuite.google.com/app/apikey")
            return False
        else:
            print("⚠️  No environment configuration found")
            print("   The application will use default settings")
    
    # Load and check environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key or gemini_key == 'your_gemini_api_key_here':
            print("⚠️  GEMINI_API_KEY not configured")
            print("   Enhanced features will not be available")
            print("   Get your API key from: https://makersuite.google.com/app/apikey")
            print("   Add it to your .env file: GEMINI_API_KEY=your_actual_key")
            return False
        
        print("✅ Environment configuration looks good")
        return True
        
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment")
        return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating necessary directories...")
    
    directories = ['logs', 'models', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")


def start_server():
    """Start the Flask application"""
    print("\n🚀 Starting the backend server...")
    print("   Server will be available at: http://localhost:5000")
    print("   API endpoints:")
    print("   - GET  /api/health          - Health check")
    print("   - POST /api/predict-salary  - Enhanced salary prediction")
    print("   - POST /api/analyze-job     - Job description analysis")
    print("   - POST /api/match-job       - Resume-job matching")
    print("\n   Press Ctrl+C to stop the server")
    print("\n" + "-"*60)
    
    try:
        # Import and run the Flask app
        from app import app
        
        # Get configuration
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Error importing application: {e}")
        print("   Please ensure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    """Main startup function"""
    print_banner()
    
    # Run all checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_requirements():
        print("\n❌ Dependency installation failed. Please resolve the issues and try again.")
        sys.exit(1)
    
    env_ok = check_environment()
    
    create_directories()
    
    if not env_ok:
        print("\n⚠️  Environment configuration issues detected.")
        response = input("   Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("   Please configure your environment and try again.")
            sys.exit(1)
    
    # Start the server
    start_server()


if __name__ == "__main__":
    main()