#!/usr/bin/env python3
"""
Gemini Semantic Extractor Module

This module provides semantic keyword extraction capabilities using Google's Gemini API.
It extracts skills, experience, and other relevant features from resumes and job descriptions
using advanced natural language processing and semantic understanding.

Features:
- Semantic skill extraction from resume text
- Experience level and domain extraction
- Job description analysis and requirement extraction
- Skills matching between resumes and job descriptions
- Confidence scoring for extracted features

Author: AI Assistant
Date: 2024
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")
    genai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedSkills:
    """
    Data class to store extracted skills and their metadata
    """
    technical_skills: List[str]
    soft_skills: List[str]
    programming_languages: List[str]
    frameworks_tools: List[str]
    certifications: List[str]
    domains: List[str]
    confidence_score: float
    extraction_timestamp: str

@dataclass
class ExperienceProfile:
    """
    Data class to store extracted experience information
    """
    years_of_experience: float
    experience_level: str  # fresher, junior, mid_level, senior
    job_roles: List[str]
    industries: List[str]
    key_achievements: List[str]
    leadership_experience: bool
    project_complexity: str  # low, medium, high
    confidence_score: float

@dataclass
class JobRequirements:
    """
    Data class to store job description requirements
    """
    required_skills: List[str]
    preferred_skills: List[str]
    experience_required: str
    job_role: str
    industry: str
    education_requirements: List[str]
    soft_skills_required: List[str]
    confidence_score: float

@dataclass
class SkillsMatchResult:
    """
    Data class to store skills matching results
    """
    overall_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    experience_match: bool
    detailed_breakdown: Dict[str, float]

class GeminiSemanticExtractor:
    """
    Main class for semantic extraction using Gemini API
    
    This class provides methods to extract semantic information from text
    using Google's Gemini API with advanced prompting techniques.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Semantic Extractor
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self._initialize_gemini()
        
        # Predefined skill categories for better extraction
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
            ],
            'frameworks_tools': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'docker',
                'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git', 'mongodb', 'postgresql'
            ],
            'domains': [
                'machine learning', 'data science', 'web development', 'mobile development',
                'devops', 'cybersecurity', 'cloud computing', 'artificial intelligence',
                'blockchain', 'iot', 'game development', 'ui/ux design'
            ]
        }
    
    def _initialize_gemini(self) -> None:
        """
        Initialize Gemini API connection
        """
        if not self.api_key:
            logger.warning("No Gemini API key provided. Some features will be limited.")
            return
            
        if genai is None:
            logger.error("google-generativeai package not installed")
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.model = None
    
    def extract_skills_from_resume(self, resume_text: str) -> ExtractedSkills:
        """
        Extract skills from resume text using semantic analysis
        
        Args:
            resume_text: The resume text to analyze
            
        Returns:
            ExtractedSkills object containing categorized skills
        """
        if not self.model:
            return self._fallback_skill_extraction(resume_text)
        
        prompt = self._create_skill_extraction_prompt(resume_text)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_skill_extraction_response(response.text)
        except Exception as e:
            logger.error(f"Error in Gemini skill extraction: {e}")
            return self._fallback_skill_extraction(resume_text)
    
    def extract_experience_profile(self, resume_text: str) -> ExperienceProfile:
        """
        Extract experience profile from resume text
        
        Args:
            resume_text: The resume text to analyze
            
        Returns:
            ExperienceProfile object containing experience information
        """
        if not self.model:
            return self._fallback_experience_extraction(resume_text)
        
        prompt = self._create_experience_extraction_prompt(resume_text)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_experience_extraction_response(response.text)
        except Exception as e:
            logger.error(f"Error in Gemini experience extraction: {e}")
            return self._fallback_experience_extraction(resume_text)
    
    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        """
        Extract requirements from job description
        
        Args:
            job_description: The job description text to analyze
            
        Returns:
            JobRequirements object containing job requirements
        """
        if not self.model:
            return self._fallback_job_requirements_extraction(job_description)
        
        prompt = self._create_job_requirements_prompt(job_description)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_job_requirements_response(response.text)
        except Exception as e:
            logger.error(f"Error in Gemini job requirements extraction: {e}")
            return self._fallback_job_requirements_extraction(job_description)
    
    def match_skills(self, resume_skills: ExtractedSkills, job_requirements: JobRequirements) -> SkillsMatchResult:
        """
        Match resume skills with job requirements
        
        Args:
            resume_skills: Extracted skills from resume
            job_requirements: Extracted requirements from job description
            
        Returns:
            SkillsMatchResult object containing matching analysis
        """
        # Combine all resume skills
        all_resume_skills = (
            resume_skills.technical_skills + 
            resume_skills.programming_languages + 
            resume_skills.frameworks_tools + 
            resume_skills.soft_skills
        )
        
        # Combine all job requirements
        all_job_skills = job_requirements.required_skills + job_requirements.preferred_skills
        
        # Calculate matches
        matched_skills = []
        missing_skills = []
        
        for job_skill in all_job_skills:
            if self._is_skill_match(job_skill, all_resume_skills):
                matched_skills.append(job_skill)
            else:
                missing_skills.append(job_skill)
        
        # Find additional skills
        additional_skills = []
        for resume_skill in all_resume_skills:
            if not self._is_skill_match(resume_skill, all_job_skills):
                additional_skills.append(resume_skill)
        
        # Calculate overall match score
        if len(all_job_skills) > 0:
            overall_match_score = len(matched_skills) / len(all_job_skills)
        else:
            overall_match_score = 0.0
        
        # Detailed breakdown
        detailed_breakdown = {
            'technical_skills_match': self._calculate_category_match(
                resume_skills.technical_skills, job_requirements.required_skills
            ),
            'programming_languages_match': self._calculate_category_match(
                resume_skills.programming_languages, job_requirements.required_skills
            ),
            'frameworks_tools_match': self._calculate_category_match(
                resume_skills.frameworks_tools, job_requirements.required_skills
            ),
            'soft_skills_match': self._calculate_category_match(
                resume_skills.soft_skills, job_requirements.soft_skills_required
            )
        }
        
        return SkillsMatchResult(
            overall_match_score=overall_match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            additional_skills=additional_skills[:10],  # Limit to top 10
            experience_match=True,  # TODO: Implement experience matching
            detailed_breakdown=detailed_breakdown
        )
    
    def _create_skill_extraction_prompt(self, resume_text: str) -> str:
        """
        Create a prompt for skill extraction
        """
        return f"""
Analyze the following resume text and extract skills in a structured format.
Focus on identifying:
1. Technical skills (specific technologies, methodologies)
2. Soft skills (communication, leadership, etc.)
3. Programming languages
4. Frameworks and tools
5. Certifications
6. Domain expertise areas

Provide the response in JSON format with the following structure:
{{
    "technical_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "programming_languages": ["language1", "language2"],
    "frameworks_tools": ["tool1", "tool2"],
    "certifications": ["cert1", "cert2"],
    "domains": ["domain1", "domain2"],
    "confidence_score": 0.85
}}

Resume Text:
{resume_text}

Response:
"""
    
    def _create_experience_extraction_prompt(self, resume_text: str) -> str:
        """
        Create a prompt for experience extraction
        """
        return f"""
Analyze the following resume text and extract experience information.
Focus on:
1. Total years of experience (calculate from job history)
2. Experience level (fresher: 0-1 years, junior: 1-3 years, mid_level: 3-7 years, senior: 7+ years)
3. Job roles held
4. Industries worked in
5. Key achievements
6. Leadership experience (yes/no)
7. Project complexity (low/medium/high based on scope and impact)

Provide the response in JSON format:
{{
    "years_of_experience": 3.5,
    "experience_level": "mid_level",
    "job_roles": ["role1", "role2"],
    "industries": ["industry1", "industry2"],
    "key_achievements": ["achievement1", "achievement2"],
    "leadership_experience": true,
    "project_complexity": "high",
    "confidence_score": 0.90
}}

Resume Text:
{resume_text}

Response:
"""
    
    def _create_job_requirements_prompt(self, job_description: str) -> str:
        """
        Create a prompt for job requirements extraction
        """
        return f"""
Analyze the following job description and extract requirements.
Identify:
1. Required skills (must-have)
2. Preferred skills (nice-to-have)
3. Experience required
4. Job role/title
5. Industry
6. Education requirements
7. Soft skills required

Provide the response in JSON format:
{{
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "experience_required": "3-5 years",
    "job_role": "Software Engineer",
    "industry": "Technology",
    "education_requirements": ["Bachelor's in CS"],
    "soft_skills_required": ["communication", "teamwork"],
    "confidence_score": 0.88
}}

Job Description:
{job_description}

Response:
"""
    
    def _parse_skill_extraction_response(self, response_text: str) -> ExtractedSkills:
        """
        Parse Gemini response for skill extraction
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ExtractedSkills(
                    technical_skills=data.get('technical_skills', []),
                    soft_skills=data.get('soft_skills', []),
                    programming_languages=data.get('programming_languages', []),
                    frameworks_tools=data.get('frameworks_tools', []),
                    certifications=data.get('certifications', []),
                    domains=data.get('domains', []),
                    confidence_score=data.get('confidence_score', 0.5),
                    extraction_timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            logger.error(f"Error parsing skill extraction response: {e}")
        
        return self._create_empty_extracted_skills()
    
    def _parse_experience_extraction_response(self, response_text: str) -> ExperienceProfile:
        """
        Parse Gemini response for experience extraction
        """
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ExperienceProfile(
                    years_of_experience=data.get('years_of_experience', 0.0),
                    experience_level=data.get('experience_level', 'fresher'),
                    job_roles=data.get('job_roles', []),
                    industries=data.get('industries', []),
                    key_achievements=data.get('key_achievements', []),
                    leadership_experience=data.get('leadership_experience', False),
                    project_complexity=data.get('project_complexity', 'low'),
                    confidence_score=data.get('confidence_score', 0.5)
                )
        except Exception as e:
            logger.error(f"Error parsing experience extraction response: {e}")
        
        return self._create_empty_experience_profile()
    
    def _parse_job_requirements_response(self, response_text: str) -> JobRequirements:
        """
        Parse Gemini response for job requirements extraction
        """
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return JobRequirements(
                    required_skills=data.get('required_skills', []),
                    preferred_skills=data.get('preferred_skills', []),
                    experience_required=data.get('experience_required', ''),
                    job_role=data.get('job_role', ''),
                    industry=data.get('industry', ''),
                    education_requirements=data.get('education_requirements', []),
                    soft_skills_required=data.get('soft_skills_required', []),
                    confidence_score=data.get('confidence_score', 0.5)
                )
        except Exception as e:
            logger.error(f"Error parsing job requirements response: {e}")
        
        return self._create_empty_job_requirements()
    
    def _fallback_skill_extraction(self, resume_text: str) -> ExtractedSkills:
        """
        Fallback skill extraction using regex patterns
        """
        text_lower = resume_text.lower()
        
        # Extract programming languages
        programming_languages = []
        for lang in self.skill_categories['programming_languages']:
            if lang in text_lower:
                programming_languages.append(lang)
        
        # Extract frameworks and tools
        frameworks_tools = []
        for tool in self.skill_categories['frameworks_tools']:
            if tool in text_lower:
                frameworks_tools.append(tool)
        
        # Extract domains
        domains = []
        for domain in self.skill_categories['domains']:
            if domain in text_lower:
                domains.append(domain)
        
        return ExtractedSkills(
            technical_skills=programming_languages + frameworks_tools,
            soft_skills=[],
            programming_languages=programming_languages,
            frameworks_tools=frameworks_tools,
            certifications=[],
            domains=domains,
            confidence_score=0.3,  # Lower confidence for fallback
            extraction_timestamp=datetime.now().isoformat()
        )
    
    def _fallback_experience_extraction(self, resume_text: str) -> ExperienceProfile:
        """
        Fallback experience extraction using regex patterns
        """
        # Simple regex to find years of experience
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*exp'
        ]
        
        years_experience = 0.0
        for pattern in experience_patterns:
            match = re.search(pattern, resume_text.lower())
            if match:
                years_experience = float(match.group(1))
                break
        
        # Categorize experience level
        if years_experience <= 1:
            experience_level = 'fresher'
        elif years_experience <= 3:
            experience_level = 'junior'
        elif years_experience <= 7:
            experience_level = 'mid_level'
        else:
            experience_level = 'senior'
        
        return ExperienceProfile(
            years_of_experience=years_experience,
            experience_level=experience_level,
            job_roles=[],
            industries=[],
            key_achievements=[],
            leadership_experience=False,
            project_complexity='medium',
            confidence_score=0.4
        )
    
    def _fallback_job_requirements_extraction(self, job_description: str) -> JobRequirements:
        """
        Fallback job requirements extraction
        """
        return JobRequirements(
            required_skills=[],
            preferred_skills=[],
            experience_required='',
            job_role='',
            industry='',
            education_requirements=[],
            soft_skills_required=[],
            confidence_score=0.2
        )
    
    def _is_skill_match(self, skill1: str, skill_list: List[str]) -> bool:
        """
        Check if a skill matches any skill in the list (case-insensitive, partial match)
        """
        skill1_lower = skill1.lower()
        for skill2 in skill_list:
            skill2_lower = skill2.lower()
            if skill1_lower in skill2_lower or skill2_lower in skill1_lower:
                return True
        return False
    
    def _calculate_category_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate match percentage for a specific skill category
        """
        if not job_skills:
            return 1.0
        
        matches = 0
        for job_skill in job_skills:
            if self._is_skill_match(job_skill, resume_skills):
                matches += 1
        
        return matches / len(job_skills)
    
    def _create_empty_extracted_skills(self) -> ExtractedSkills:
        """
        Create empty ExtractedSkills object
        """
        return ExtractedSkills(
            technical_skills=[],
            soft_skills=[],
            programming_languages=[],
            frameworks_tools=[],
            certifications=[],
            domains=[],
            confidence_score=0.0,
            extraction_timestamp=datetime.now().isoformat()
        )
    
    def _create_empty_experience_profile(self) -> ExperienceProfile:
        """
        Create empty ExperienceProfile object
        """
        return ExperienceProfile(
            years_of_experience=0.0,
            experience_level='fresher',
            job_roles=[],
            industries=[],
            key_achievements=[],
            leadership_experience=False,
            project_complexity='low',
            confidence_score=0.0
        )
    
    def _create_empty_job_requirements(self) -> JobRequirements:
        """
        Create empty JobRequirements object
        """
        return JobRequirements(
            required_skills=[],
            preferred_skills=[],
            experience_required='',
            job_role='',
            industry='',
            education_requirements=[],
            soft_skills_required=[],
            confidence_score=0.0
        )
    
    def save_extraction_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save extraction results to JSON file
        
        Args:
            results: Dictionary containing extraction results
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_extraction_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load extraction results from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary containing extraction results or None if error
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

# Example usage and testing functions
def test_gemini_extractor():
    """
    Test function for the Gemini Semantic Extractor
    """
    extractor = GeminiSemanticExtractor()
    
    # Sample resume text
    sample_resume = """
    John Doe
    Software Engineer with 5 years of experience in Python, JavaScript, and React.
    Worked at Tech Corp as Senior Developer, leading a team of 3 developers.
    Built scalable web applications using Django and PostgreSQL.
    Certified in AWS Cloud Practitioner.
    Strong communication and leadership skills.
    """
    
    # Sample job description
    sample_job = """
    We are looking for a Senior Software Engineer with 4+ years of experience.
    Required skills: Python, React, PostgreSQL, AWS
    Preferred skills: Docker, Kubernetes
    Must have strong communication skills and leadership experience.
    Bachelor's degree in Computer Science required.
    """
    
    # Extract skills and experience
    skills = extractor.extract_skills_from_resume(sample_resume)
    experience = extractor.extract_experience_profile(sample_resume)
    job_reqs = extractor.extract_job_requirements(sample_job)
    
    # Match skills
    match_result = extractor.match_skills(skills, job_reqs)
    
    print("Extracted Skills:", asdict(skills))
    print("Experience Profile:", asdict(experience))
    print("Job Requirements:", asdict(job_reqs))
    print("Skills Match:", asdict(match_result))

if __name__ == "__main__":
    test_gemini_extractor()