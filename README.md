# ATS Resume Analyzer - Enhanced with Gemini AI

## Overview

The ATS Resume Analyzer is an advanced web application that combines traditional resume analysis with cutting-edge AI technology. This enhanced version integrates Google's Gemini API for semantic keyword extraction and provides sophisticated salary prediction and job matching capabilities.

## Features

### 🎯 Core Features
- **Resume Analysis**: Upload and analyze resumes in PDF, DOC, or text format
- **Enhanced Salary Prediction**: AI-powered salary estimation using semantic analysis
- **Job Description Matching**: Match resumes against job requirements
- **Skills Extraction**: Identify technical and soft skills using Gemini AI
- **Experience Analysis**: Evaluate years of experience and expertise levels

### 🚀 Enhanced Features (v2.0)
- **Gemini AI Integration**: Semantic keyword extraction using sentence transformers
- **Advanced ML Model**: Retrained with semantically extracted features
- **Job Matching Interface**: HR-friendly job description input and analysis
- **Real-time Matching**: Live skills matching with recommendations
- **RESTful API**: Backend services for enhanced functionality

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Gemini AI     │
│   (HTML/JS)     │◄──►│   (Flask)        │◄──►│   Service       │
│                 │    │                  │    │                 │
│ • Resume Upload │    │ • Salary Predict │    │ • Semantic      │
│ • Job Matching  │    │ • Job Analysis   │    │   Extraction    │
│ • Results UI    │    │ • Skills Match   │    │ • Skill ID      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   ML Models      │
                       │                  │
                       │ • Salary Model   │
                       │ • Feature Eng.   │
                       │ • Predictions    │
                       └──────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Modern web browser

### Quick Start

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd ats-resume-analyzer
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file and add your Gemini API key
   # Get your key from: https://makersuite.google.com/app/apikey
   ```

3. **Start the backend server**
   ```bash
   python start_backend.py
   ```
   This script will:
   - Check Python version compatibility
   - Install required dependencies
   - Validate environment configuration
   - Start the Flask backend server

4. **Start the frontend server**
   ```bash
   # In a new terminal
   python -m http.server 8080
   ```

5. **Access the application**
   - Frontend: http://localhost:8080
   - Backend API: http://localhost:5000

### Manual Installation

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY="your_api_key_here"

# Start backend
python app.py

# Start frontend (in another terminal)
python -m http.server 8080
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
FLASK_PORT=5000
FLASK_DEBUG=True
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:8080
```

### API Configuration

- **Rate Limiting**: 60 requests per minute (configurable)
- **Cache Timeout**: 1 hour for repeated requests
- **Max File Size**: 10MB for resume uploads
- **Supported Formats**: PDF, DOC, DOCX, TXT

## Usage

### Resume Analysis

1. **Upload Resume**: Drag and drop or click to upload
2. **Analysis**: The system extracts:
   - Personal information
   - Skills (technical and soft)
   - Experience details
   - Education background
   - Certifications

3. **Salary Prediction**: Get enhanced predictions based on:
   - Semantic skill analysis
   - Experience evaluation
   - Market data correlation
   - Location and industry factors

### Job Description Matching

1. **Switch to Job Matching Tab**
2. **Enter Job Description**: Paste the complete job posting
3. **Analyze Requirements**: System extracts:
   - Required skills
   - Preferred qualifications
   - Experience requirements
   - Education needs

4. **Upload Resume for Matching**
5. **View Results**:
   - Overall match percentage
   - Matched skills
   - Missing skills
   - Additional skills
   - Improvement recommendations

## API Endpoints

### Health Check
```http
GET /api/health
```
Returns service status and component health.

### Enhanced Salary Prediction
```http
POST /api/predict-salary
Content-Type: application/json

{
  "resume_text": "string",
  "job_title": "string" (optional),
  "location": "string" (optional),
  "company_size": "string" (optional)
}
```

### Job Description Analysis
```http
POST /api/analyze-job
Content-Type: application/json

{
  "job_description": "string"
}
```

### Resume-Job Matching
```http
POST /api/match-job
Content-Type: application/json

{
  "job_description": "string",
  "resume_text": "string"
}
```

## File Structure

```
ats-resume-analyzer/
├── index.html                 # Main frontend application
├── salary-prediction.js       # Enhanced frontend logic
├── app.py                     # Flask backend server
├── start_backend.py           # Startup script
├── enhanced_salary_predictor.py # Enhanced ML predictor
├── gemini_semantic_extractor.py # Gemini AI integration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .env                      # Your configuration (create this)
├── README.md                 # This file
├── logs/                     # Application logs
├── models/                   # ML model storage
└── temp/                     # Temporary file storage
```

## Development

### Adding New Features

1. **Backend**: Extend `app.py` with new endpoints
2. **AI Integration**: Modify `gemini_semantic_extractor.py`
3. **ML Models**: Update `enhanced_salary_predictor.py`
4. **Frontend**: Enhance `salary-prediction.js` and `index.html`

### Testing

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest

# Code formatting
black .

# Linting
flake8 .
```

### Logging

Logs are stored in the `logs/` directory:
- `app.log`: Application logs with rotation
- Console output: Colored logs for development

## Troubleshooting

### Common Issues

1. **"Gemini API key not configured"**
   - Ensure your `.env` file contains a valid `GEMINI_API_KEY`
   - Get your key from: https://makersuite.google.com/app/apikey

2. **"Service not available"**
   - Check if the backend server is running on port 5000
   - Verify CORS configuration in `.env`

3. **"Dependencies installation failed"**
   - Ensure Python 3.8+ is installed
   - Try: `pip install --upgrade pip`
   - Use virtual environment: `python -m venv venv && source venv/bin/activate`

4. **"Frontend not loading"**
   - Ensure frontend server is running on port 8080
   - Check browser console for errors
   - Verify file paths and permissions

### Performance Optimization

- **Caching**: Enable Redis for production deployments
- **Rate Limiting**: Adjust API limits based on usage
- **Model Optimization**: Retrain models with more data
- **CDN**: Use CDN for static assets in production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on the repository

## Changelog

### Version 2.0.0
- ✨ Gemini AI integration for semantic analysis
- 🚀 Enhanced salary prediction with ML improvements
- 💼 Job description matching functionality
- 🎨 Improved UI with job matching interface
- 🔧 RESTful API backend
- 📊 Advanced skills extraction and analysis

### Version 1.0.0
- 📄 Basic resume analysis
- 💰 Simple salary prediction
- 🎯 ATS scoring system
- 📱 Responsive web interface