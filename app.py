#!/usr/bin/env python3
"""
ATS Resume Analyzer - Enhanced Backend Server

This is the main Flask application that serves as the backend for the ATS Resume Analyzer.
It provides enhanced salary prediction using Gemini API for semantic keyword extraction
and job description matching capabilities.

Features:
- Enhanced salary prediction with semantic analysis
- Job description matching and skills analysis
- RESTful API endpoints
- CORS support for frontend integration
- Comprehensive logging and error handling

Author: AI Assistant
Date: 2024
"""

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from loguru import logger
import sys

# Import our custom modules
# Removed salary prediction imports

# Load environment variables
load_dotenv()

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    os.getenv('LOG_FILE', 'logs/app.log'),
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:8080').split(',')
CORS(app, origins=cors_origins)

# Initialize our services
# Removed salary prediction initialization


@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service status
    
    Returns:
        JSON response with service status and component health
    """
    try:
        status = {
            'status': 'healthy',
            'services': {
                'api_version': '1.0.0',
                'ats_analyzer': True
            },
            'version': '1.0.0'
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# Removed salary prediction endpoint


@app.route('/api/match-job', methods=['POST'])
def match_job_description():
    """
    Job description matching endpoint
    
    Expected JSON payload:
    {
        "job_description": "string",
        "resume_text": "string"
    }
    
    Returns:
        JSON response with matching analysis and recommendations
    """
    try:
        if not gemini_extractor:
            return jsonify({
                'error': 'Job matching service not available',
                'fallback': True
            }), 503
        
        data = request.get_json()
        if not data or 'job_description' not in data or 'resume_text' not in data:
            return jsonify({
                'error': 'Both job_description and resume_text are required'
            }), 400
        
        logger.info("Processing job matching request")
        
        # Extract job requirements
        job_requirements = gemini_extractor.extract_job_requirements(
            data['job_description']
        )
        
        # Extract resume skills and experience
        resume_skills = gemini_extractor.extract_skills_from_resume(data['resume_text'])
        resume_experience = gemini_extractor.extract_experience_profile(data['resume_text'])
        
        # Perform matching
        match_result = gemini_extractor.match_skills(
            resume_skills, job_requirements
        )
        
        # Return job matching analysis in expected format
        result = {
            'overall_match_score': match_result.overall_match_score * 100,
            'skills_analysis': {
                'matched_skills': match_result.matched_skills,
                'missing_skills': match_result.missing_skills,
                'additional_skills': match_result.additional_skills,
                'detailed_breakdown': match_result.detailed_breakdown
            },
            'job_analysis': {
                'required_skills': job_requirements.required_skills,
                'preferred_skills': job_requirements.preferred_skills,
                'experience_required': job_requirements.experience_required,
                'education_requirements': job_requirements.education_requirements
            },
            'resume_analysis': {
                'technical_skills': resume_skills.technical_skills,
                'soft_skills': resume_skills.soft_skills,
                'certifications': resume_skills.certifications,
                'years_experience': resume_experience.years_of_experience,
                'experience_level': resume_experience.experience_level
            }
        }
        
        logger.info(f"Job matching completed with score: {match_result.overall_match_score * 100:.1f}%")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Job matching error: {e}")
        return jsonify({
            'error': 'Internal server error during job matching',
            'details': str(e)
        }), 500


@app.route('/api/analyze-job', methods=['POST'])
def analyze_job_description():
    """
    Analyze job description only (without resume matching)
    
    Expected JSON payload:
    {
        "job_description": "string"
    }
    
    Returns:
        JSON response with job analysis results
    """
    try:
        if not gemini_extractor:
            return jsonify({
                'error': 'Job analysis service not available',
                'fallback': True
            }), 503
        
        data = request.get_json()
        if not data or 'job_description' not in data:
            return jsonify({'error': 'job_description is required'}), 400
        
        logger.info("Processing job analysis request")
        
        # Extract job requirements
        job_requirements = gemini_extractor.extract_job_requirements(
            data['job_description']
        )
        
        result = {
            'required_skills': job_requirements.required_skills,
            'preferred_skills': job_requirements.preferred_skills,
            'experience_requirements': job_requirements.experience_required,
            'education_required': job_requirements.education_requirements,
            'job_role': job_requirements.job_role,
            'industry': job_requirements.industry,
            'soft_skills_required': job_requirements.soft_skills_required,
            'confidence_score': job_requirements.confidence_score
        }
        
        logger.info("Job analysis completed successfully")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Job analysis error: {e}")
        return jsonify({
            'error': 'Internal server error during job analysis',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get configuration from environment
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting ATS Resume Analyzer Backend Server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CORS origins: {cors_origins}")
    
    # Start the Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )