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
from salary_calculator import OTSSalaryCalculator
from gemini_semantic_extractor import GeminiSemanticExtractor

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
cors_origins = os.getenv('CORS_ORIGINS', '*')
if cors_origins == '*':
    CORS(app)
else:
    CORS(app, origins=cors_origins.split(','))

# Initialize our services
try:
    salary_calculator = OTSSalaryCalculator('ots_salary_config.json')
    logger.info("OTS Salary Calculator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize salary calculator: {e}")
    salary_calculator = None

try:
    gemini_extractor = GeminiSemanticExtractor()
    logger.info("Gemini Semantic Extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini extractor: {e}")
    gemini_extractor = None


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
                'ats_analyzer': True,
                'salary_calculator': salary_calculator is not None
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


@app.route('/api/predict-salary', methods=['POST'])
def predict_salary():
    """
    Predict salary based on resume analysis using OTS salary calculator
    
    Expected JSON payload:
    {
        "resume_text": "string"
    }
    
    Returns:
        JSON response with salary prediction and detailed breakdown
    """
    try:
        if not salary_calculator:
            return jsonify({
                'error': 'Salary prediction service not available',
                'fallback': True
            }), 503
        
        data = request.get_json()
        if not data or 'resume_text' not in data:
            return jsonify({
                'error': 'resume_text is required'
            }), 400
        
        logger.info("Processing salary prediction request")
        
        # Calculate salary using OTS calculator
        result = salary_calculator.calculate_salary(data['resume_text'])
        
        # Check for validation errors (non-resume documents)
        if result.breakdown.get('validation_error', False):
            logger.warning(f"Document validation failed: {result.breakdown.get('error_message', 'Unknown validation error')}")
            return jsonify({
                'error': 'Invalid document type',
                'validation_failed': True,
                'document_analysis': {
                    'document_type': result.breakdown.get('document_type', 'unknown'),
                    'validation_result': result.breakdown.get('validation_result', 'invalid'),
                    'confidence_score': result.breakdown.get('confidence_score', 0.0),
                    'quality_score': result.breakdown.get('quality_score', 0.0),
                    'recommendations': result.breakdown.get('recommendations', []),
                    'message': result.breakdown.get('error_message', 'Document does not appear to be a valid resume')
                },
                'suggestions': [
                    'Please upload a valid resume document',
                    'Ensure the document contains typical resume sections like education, experience, and skills',
                    'Check that the document is not an invoice, report, or other non-resume document'
                ]
            }), 400
        
        # Project salary growth
        growth_projection_raw = salary_calculator.project_salary_growth(
            result.final_salary, years=5, 
            current_experience=result.breakdown['years_experience'],
            current_role=result.breakdown['role']
        )
        
        # Format growth projection for frontend
        growth_projection = {
            'year_1': {
                'salary': growth_projection_raw['yearly_breakdown'].get('year_1', {}).get('salary', result.final_salary),
                'percentage': growth_projection_raw['yearly_breakdown'].get('year_1', {}).get('total_growth_from_current', 0)
            },
            'year_3': {
                'salary': growth_projection_raw['yearly_breakdown'].get('year_3', {}).get('salary', result.final_salary),
                'percentage': growth_projection_raw['yearly_breakdown'].get('year_3', {}).get('total_growth_from_current', 0)
            },
            'year_5': {
                'salary': growth_projection_raw['yearly_breakdown'].get('year_5', {}).get('salary', result.final_salary),
                'percentage': growth_projection_raw['yearly_breakdown'].get('year_5', {}).get('total_growth_from_current', 0)
            },
            'role_transitions': growth_projection_raw.get('role_transitions', []),
            'growth_factors': growth_projection_raw.get('growth_factors', {})
        }
        
        # Calculate dynamic salary range
        dynamic_range = salary_calculator.calculate_dynamic_salary_range(
            base_salary=result.base_salary,
            experience_band=result.experience_band,
            skills=result.breakdown.get('skills', []),
            location=result.breakdown.get('location', ''),
            college_tier=result.college_tier,
            college_multiplier=result.college_multiplier
        )
        
        # Format response with enhanced college tier information
        response = {
            'estimated_salary': result.final_salary,
            'experience_level': result.experience_band,
            'salary_range': {
                'min': dynamic_range['min'],
                'max': dynamic_range['max'],
                'median': dynamic_range['median'],
                'confidence_level': dynamic_range['confidence_level'],
                'market_factors': dynamic_range['market_factors'],
                'recommendations': dynamic_range['recommendations']
            },
            'breakdown': {
                'base_salary': result.base_salary,
                'college_premium': (result.final_salary - result.base_salary) if result.college_multiplier > 1.0 else 0,
                'role_adjustment': 0,  # Can be enhanced later
                'location_factor': result.location_multiplier,
                'college_tier': {
                    'tier': result.college_tier,
                    'institution': result.breakdown.get('college_name', 'Unknown')
                } if result.college_tier else None,
                'college_multiplier': result.college_multiplier,
                'years_experience': result.breakdown['years_experience'],
                'college_name': result.breakdown['college_name'],
                'graduation_year': result.breakdown['graduation_year'],
                'identified_role': result.breakdown['role'],
                'location': result.breakdown['location'],
                'skills_identified': result.breakdown['skills']
            },
            'calculation_details': {
                'premium_college_info': result.breakdown.get('premium_college_info', {}),
                'multipliers': {
                    'college_premium': result.college_multiplier,
                    'role_adjustment': result.role_multiplier,
                    'location_adjustment': result.location_multiplier,
                    'skill_bonus': result.skill_multiplier
                }
            },
            'growth_projection': growth_projection,
            'confidence_factors': {
                'experience_confidence': 'high' if result.breakdown['years_experience'] > 0 else 'low',
                'college_confidence': 'high' if result.breakdown['college_name'] else 'medium',
                'role_confidence': 'high' if result.breakdown['role'] != 'backend_developer' else 'medium',
                'overall_confidence': 'high' if all([
                    result.breakdown['years_experience'] > 0,
                    result.breakdown['college_name'],
                    len(result.breakdown['skills']) > 2
                ]) else 'medium'
            }
        }
        
        logger.info(f"Salary prediction completed: ₹{result.final_salary} LPA for {result.experience_band} level")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Salary prediction error: {e}")
        return jsonify({
            'error': 'Internal server error during salary prediction',
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