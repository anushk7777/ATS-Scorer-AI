# ATS Resume Analyzer

## Overview

The ATS Resume Analyzer is an advanced web application that provides comprehensive resume analysis and scoring. This application helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) by analyzing format, content quality, keyword optimization, and overall ATS readability.

## Features

### 🎯 Core Features
- **Resume Analysis**: Upload and analyze resumes in PDF, DOC, or text format
<<<<<<< HEAD
- **ATS Scoring**: Comprehensive scoring system for ATS compatibility
- **Format Analysis**: Evaluate resume structure and formatting
- **Content Quality**: Analyze content depth and relevance
- **Keyword Optimization**: Identify keyword usage and optimization opportunities
- **ATS Readability**: Assess how well ATS systems can parse the resume
- **Detailed Feedback**: Get specific recommendations for improvement
- **Modern UI**: Beautiful, responsive interface with glassmorphism design

## Technical Stack

### Frontend Technologies
- **HTML5**: Semantic markup and modern web standards
- **CSS3**: Advanced styling with glassmorphism effects and animations
- **JavaScript (ES6+)**: Interactive functionality and resume analysis
- **File API**: Resume upload and processing capabilities
- **Canvas API**: Dynamic score visualizations and charts
- **Local Storage**: Client-side data persistence

### Analysis Engine
- **Text Processing**: Advanced content analysis algorithms
- **Format Detection**: Resume structure and layout evaluation
- **Keyword Analysis**: ATS-friendly keyword optimization
- **Scoring System**: Comprehensive ATS compatibility scoring
- **Report Generation**: Detailed feedback and recommendations
=======
- **Enhanced Salary Prediction**: AI-powered salary estimation using semantic analysis
- **Skills Extraction**: Identify technical and soft skills using Gemini AI
- **Experience Analysis**: Evaluate years of experience and expertise levels

### 🚀 Enhanced Features (v2.0)
- **Gemini AI Integration**: Semantic keyword extraction using sentence transformers
- **Advanced ML Model**: Retrained with semantically extracted features
- **Real-time Matching**: Live skills matching with recommendations
- **RESTful API**: Backend services for enhanced functionality
>>>>>>> 1aafc1989f0055cf7614dc9e9136fbc99af31dc5

## Architecture

```
<<<<<<< HEAD
┌─────────────────────────────────────────┐
│              Frontend Application        │
│              (HTML/CSS/JS)              │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Upload    │  │   Analysis      │   │
│  │   System    │  │   Engine        │   │
│  │             │  │                 │   │
│  │ • File      │  │ • Format Check  │   │
│  │   Handling  │  │ • Content Eval  │   │
│  │ • Validation│  │ • Keyword Scan  │   │
│  │ • Preview   │  │ • ATS Scoring   │   │
│  └─────────────┘  └─────────────────┘   │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │          Results Display            │ │
│  │                                     │ │
│  │ • Score Visualization               │ │
│  │ • Detailed Analysis                 │ │
│  │ • Improvement Suggestions           │ │
│  │ • Report Generation                 │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
=======
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Gemini AI     │
│   (HTML/JS)     │◄──►│   (Flask)        │◄──►│   Service       │
│                 │    │                  │    │                 │
│ • Resume Upload │    │ • Salary Predict │    │ • Semantic      │

│ • Results UI    │    │ • Skills Match   │    │ • Skill ID      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   ML Models      │
                       │                  │
            
                       │ • Feature Eng.   │
                       │ • Predictions    │
                       └──────────────────┘
>>>>>>> 1aafc1989f0055cf7614dc9e9136fbc99af31dc5
```

## Installation & Setup

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local web server (optional, for file upload functionality)

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ats-resume-analyzer.git
   cd ats-resume-analyzer
   ```

2. **Start a local server** (recommended)
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   
   # Using PHP
   php -S localhost:8000
   ```

3. **Open the application**
   - Navigate to `http://localhost:8000`
   - Or open `index.html` directly in your browser

### Usage
1. **Upload a resume**
   - Drag and drop or click to select a resume file
   - Supported formats: PDF, DOC, DOCX, TXT

2. **Analyze results**
   - View comprehensive ATS compatibility analysis
   - Get detailed scoring and feedback
   - Download analysis reports
   - Use "Analyze Another Resume" to test multiple resumes

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
