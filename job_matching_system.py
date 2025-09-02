#!/usr/bin/env python3
"""
ATS Resume Analyzer - Advanced Job Matching System

This module implements a sophisticated job matching system that uses semantic similarity
to match resumes against job descriptions. It provides detailed analysis of skill gaps,
experience alignment, and generates actionable recommendations.

Features:
- Semantic similarity matching using Gemini AI
- Multi-dimensional scoring (skills, experience, education)
- Skill gap analysis and recommendations
- Experience level alignment
- Industry-specific matching
- Confidence scoring and explanation

Author: AI Assistant
Date: 2024
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import Counter
import difflib
from loguru import logger

# Import sentence transformers for enhanced semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: sentence-transformers not available ({e}). Using fallback similarity methods.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Import our custom modules
from gemini_semantic_extractor import (
    GeminiSemanticExtractor, 
    ExtractedSkills, 
    ExperienceProfile, 
    JobRequirements,
    SkillsMatchResult
)


@dataclass
class MatchingWeights:
    """
    Configuration for matching algorithm weights.
    """
    skills_weight: float = 0.6
    experience_weight: float = 0.3
    education_weight: float = 0.1
    technical_skills_weight: float = 0.7
    soft_skills_weight: float = 0.2
    certifications_weight: float = 0.1


@dataclass
class DetailedMatchResult:
    """
    Comprehensive result of job-resume matching analysis.
    """
    overall_score: float
    skills_score: float
    experience_score: float
    education_score: float
    
    # Skill analysis
    matched_technical_skills: List[str]
    matched_soft_skills: List[str]
    matched_certifications: List[str]
    missing_required_skills: List[str]
    missing_preferred_skills: List[str]
    additional_skills: List[str]
    
    # Experience analysis
    experience_gap: float  # Years difference (negative if under-qualified)
    experience_alignment: str  # "perfect", "good", "acceptable", "insufficient"
    relevant_experience: List[str]
    
    # Education analysis
    education_match: bool
    education_level_comparison: str
    
    # Recommendations
    improvement_recommendations: List[str]
    strength_highlights: List[str]
    interview_talking_points: List[str]
    
    # Metadata
    confidence_score: float
    analysis_timestamp: str
    match_explanation: str


class AdvancedJobMatcher:
    """
    Advanced job matching system with semantic analysis capabilities.
    """
    
    def __init__(self, gemini_extractor: GeminiSemanticExtractor = None, 
                 weights: MatchingWeights = None):
        """
        Initialize the advanced job matcher with enhanced semantic matching capabilities.
        
        Args:
            gemini_extractor: Instance of GeminiSemanticExtractor
            weights: Custom weights for matching algorithm
        """
        self.gemini_extractor = gemini_extractor or GeminiSemanticExtractor()
        self.weights = weights or MatchingWeights()
        
        # Initialize sentence transformer model for enhanced semantic similarity
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a model optimized for semantic similarity
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        
        # Skill similarity thresholds
        self.similarity_thresholds = {
            'exact_match': 0.95,
            'high_similarity': 0.8,
            'moderate_similarity': 0.6,
            'low_similarity': 0.4
        }
        
        # Experience level mappings
        self.experience_levels = {
            'entry': (0, 2),
            'junior': (1, 3),
            'mid': (2, 5),
            'senior': (5, 10),
            'lead': (7, 15),
            'principal': (10, 20)
        }
        
        logger.info("Advanced Job Matcher initialized with enhanced semantic matching")
    
    def match_resume_to_job(self, resume_text: str, job_description: str) -> DetailedMatchResult:
        """
        Perform comprehensive matching between resume and job description.
        
        Args:
            resume_text: Full text of the resume
            job_description: Full text of the job description
            
        Returns:
            DetailedMatchResult with comprehensive analysis
        """
        logger.info("Starting comprehensive job-resume matching")
        
        try:
            # Extract features from both resume and job description
            resume_skills = self.gemini_extractor.extract_skills(resume_text)
            resume_experience = self.gemini_extractor.extract_experience(resume_text)
            job_requirements = self.gemini_extractor.extract_job_requirements(job_description)
            
            # Perform detailed skill matching
            skill_analysis = self._analyze_skills_match(
                resume_skills, job_requirements
            )
            
            # Analyze experience alignment
            experience_analysis = self._analyze_experience_match(
                resume_experience, job_requirements
            )
            
            # Analyze education requirements
            education_analysis = self._analyze_education_match(
                resume_text, job_requirements
            )
            
            # Calculate overall scores using enhanced analysis
            skills_score = self._calculate_skills_score(skill_analysis)
            experience_score = self._calculate_experience_score(experience_analysis)
            education_score = self._calculate_education_score(education_analysis)
            
            # Calculate overall score with weights and confidence adjustments
            base_overall_score = (
                skills_score * self.weights.skills_weight +
                experience_score * self.weights.experience_weight +
                education_score * self.weights.education_weight
            )
            
            # Apply confidence-based adjustments
            confidence_scores = skill_analysis.get('confidence_scores', {})
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.7
            
            # Adjust overall score based on confidence
            overall_score = base_overall_score * (0.8 + 0.2 * avg_confidence)
            
            # Generate enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(
                skill_analysis, experience_analysis, education_analysis
            )
            
            # Calculate confidence score with semantic similarity consideration
            confidence_score = self._calculate_enhanced_confidence_score(
                skills_score, experience_score, education_score, confidence_scores
             )
             
             # Create comprehensive result
            result = DetailedMatchResult(
                overall_score=round(overall_score, 2),
                skills_score=round(skills_score, 2),
                experience_score=round(experience_score, 2),
                education_score=round(education_score, 2),
                
                matched_technical_skills=skill_analysis['matched_technical'],
                matched_soft_skills=skill_analysis['matched_soft'],
                matched_certifications=skill_analysis['matched_certifications'],
                missing_required_skills=skill_analysis['missing_required'],
                missing_preferred_skills=skill_analysis['missing_preferred'],
                additional_skills=skill_analysis['additional_skills'],
                
                experience_gap=experience_analysis['gap_years'],
                experience_alignment=experience_analysis['alignment'],
                relevant_experience=experience_analysis['relevant_areas'],
                
                education_match=education_analysis['meets_requirements'],
                education_level_comparison=education_analysis['level_comparison'],
                
                improvement_recommendations=recommendations['improvements'],
                strength_highlights=recommendations['strengths'],
                interview_talking_points=recommendations['talking_points'],
                
                confidence_score=self._calculate_confidence_score(
                    resume_skills, resume_experience, job_requirements
                ),
                analysis_timestamp=datetime.now().isoformat(),
                match_explanation=self._generate_match_explanation(
                    overall_score, skill_analysis, experience_analysis
                )
            )
            
            logger.info(f"Job matching completed. Overall score: {overall_score:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in job matching: {e}")
            raise
    
    def _analyze_skills_match(self, resume_skills: ExtractedSkills, 
                             job_requirements: JobRequirements) -> Dict[str, Any]:
        """
        Analyze skill matching between resume and job requirements using enhanced semantic matching.
        
        Args:
            resume_skills: Extracted skills from resume
            job_requirements: Extracted requirements from job description
            
        Returns:
            Dictionary with detailed skill analysis including confidence scores
        """
        # Combine all resume skills
        all_resume_skills = (
            resume_skills.technical_skills + 
            resume_skills.soft_skills + 
            resume_skills.certifications
        )
        
        # Get detailed matching for required skills
        required_skills_analysis = self.get_detailed_skill_matching(
            all_resume_skills, job_requirements.required_skills
        )
        
        # Get detailed matching for preferred skills
        preferred_skills_analysis = self.get_detailed_skill_matching(
            all_resume_skills, job_requirements.preferred_skills
        )
        
        # Get detailed matching for soft skills
        soft_skills_analysis = self.get_detailed_skill_matching(
            resume_skills.soft_skills, job_requirements.soft_skills_required
        )
        
        # Extract matched skills by category
        matched_technical = [
            match['resume_skill'] for match in required_skills_analysis['matched_skills']
            if match['resume_skill'] in resume_skills.technical_skills + resume_skills.programming_languages + resume_skills.frameworks_tools
        ]
        
        matched_soft = [
            match['resume_skill'] for match in soft_skills_analysis['matched_skills']
        ]
        
        matched_certifications = [
            skill for skill in resume_skills.certifications
            if self._is_skill_covered(skill, job_requirements.required_skills + job_requirements.preferred_skills)
        ]
        
        # Calculate skill coverage percentages
        required_coverage = required_skills_analysis['overall_similarity']
        preferred_coverage = preferred_skills_analysis['overall_similarity']
        soft_skills_coverage = soft_skills_analysis['overall_similarity']
        
        return {
            'matched_technical': matched_technical,
            'matched_soft': matched_soft,
            'matched_certifications': matched_certifications,
            'missing_required': required_skills_analysis['missing_skills'],
            'missing_preferred': preferred_skills_analysis['missing_skills'],
            'additional_skills': required_skills_analysis['additional_skills'],
            'total_required_skills': len(job_requirements.required_skills),
            'total_preferred_skills': len(job_requirements.preferred_skills),
            'total_matched_skills': len(required_skills_analysis['matched_skills']),
            'required_skills_coverage': required_coverage,
            'preferred_skills_coverage': preferred_coverage,
            'detailed_required_matches': required_skills_analysis['matched_skills'],
            'detailed_preferred_matches': preferred_skills_analysis['matched_skills'],
            'confidence_scores': {
                'required_skills': self._calculate_average_confidence(required_skills_analysis['matched_skills']),
                'preferred_skills': self._calculate_average_confidence(preferred_skills_analysis['matched_skills'])
            }
        }
    
    def _find_skill_matches(self, resume_skills: List[str], 
                           job_skills: List[str]) -> List[str]:
        """
        Find matching skills using enhanced semantic similarity with batch processing.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            List of matched skills with their best matches
        """
        if not resume_skills or not job_skills:
            return []
        
        matches = []
        
        # Use batch similarity calculation for better performance
        if len(resume_skills) > 5 and len(job_skills) > 5:
            try:
                similarity_matrix = self._calculate_batch_skill_similarity(resume_skills, job_skills)
                
                # Find best matches for each resume skill
                for i, resume_skill in enumerate(resume_skills):
                    max_similarity = np.max(similarity_matrix[i])
                    if max_similarity >= self.similarity_thresholds['moderate_similarity']:
                        matches.append(resume_skill)
                        
            except Exception as e:
                logger.warning(f"Error in batch skill matching: {e}")
                # Fallback to individual matching
                return self._find_skill_matches_individual(resume_skills, job_skills)
        else:
            # Use individual matching for smaller lists
            return self._find_skill_matches_individual(resume_skills, job_skills)
        
        return matches
    
    def _find_skill_matches_individual(self, resume_skills: List[str], 
                                     job_skills: List[str]) -> List[str]:
        """
        Find matching skills using individual similarity calculations.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            List of matched skills
        """
        matches = []
        
        for resume_skill in resume_skills:
            for job_skill in job_skills:
                similarity = self._calculate_skill_similarity(resume_skill, job_skill)
                if similarity >= self.similarity_thresholds['moderate_similarity']:
                    matches.append(resume_skill)
                    break
        
        return matches
    
    def get_detailed_skill_matching(self, resume_skills: List[str], 
                                  job_skills: List[str]) -> Dict[str, Any]:
        """
        Get detailed skill matching analysis with confidence scores and similarity details.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            Dictionary containing detailed matching analysis
        """
        if not resume_skills or not job_skills:
            return {
                'matched_skills': [],
                'missing_skills': job_skills,
                'additional_skills': resume_skills,
                'match_details': [],
                'overall_similarity': 0.0
            }
        
        match_details = []
        matched_skills = []
        missing_skills = []
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_batch_skill_similarity(resume_skills, job_skills)
        
        # Analyze each job skill requirement
        for j, job_skill in enumerate(job_skills):
            best_match_idx = np.argmax(similarity_matrix[:, j])
            best_similarity = similarity_matrix[best_match_idx, j]
            
            if best_similarity >= self.similarity_thresholds['moderate_similarity']:
                matched_skills.append({
                    'job_skill': job_skill,
                    'resume_skill': resume_skills[best_match_idx],
                    'similarity': float(best_similarity),
                    'confidence': self._get_confidence_level(best_similarity)
                })
            else:
                missing_skills.append(job_skill)
        
        # Find additional skills (resume skills not matched to any job skill)
        matched_resume_indices = set()
        for match in matched_skills:
            for i, resume_skill in enumerate(resume_skills):
                if resume_skill == match['resume_skill']:
                    matched_resume_indices.add(i)
                    break
        
        additional_skills = [
            resume_skills[i] for i in range(len(resume_skills)) 
            if i not in matched_resume_indices
        ]
        
        # Calculate overall similarity
        overall_similarity = len(matched_skills) / len(job_skills) if job_skills else 0.0
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'additional_skills': additional_skills,
            'match_details': match_details,
            'overall_similarity': overall_similarity,
            'total_job_skills': len(job_skills),
            'total_resume_skills': len(resume_skills),
            'match_count': len(matched_skills)
        }
    
    def _get_confidence_level(self, similarity: float) -> str:
        """
        Get confidence level based on similarity score.
        
        Args:
            similarity: Similarity score between 0 and 1
            
        Returns:
            Confidence level string
        """
        if similarity >= self.similarity_thresholds['high_similarity']:
            return 'high'
        elif similarity >= self.similarity_thresholds['moderate_similarity']:
            return 'medium'
        elif similarity >= self.similarity_thresholds['low_similarity']:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate semantic similarity between two skills using enhanced methods.
        
        Args:
            skill1: First skill
            skill2: Second skill
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize skills
        skill1_norm = skill1.lower().strip()
        skill2_norm = skill2.lower().strip()
        
        # Exact match
        if skill1_norm == skill2_norm:
            return 1.0
        
        # Check if one skill contains the other
        if skill1_norm in skill2_norm or skill2_norm in skill1_norm:
            return 0.9
        
        # Use sentence transformers for semantic similarity if available
        if self.sentence_model is not None:
            try:
                embeddings = self.sentence_model.encode([skill1_norm, skill2_norm])
                # Calculate cosine similarity
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                # Ensure similarity is between 0 and 1
                similarity = max(0, min(1, similarity))
            except Exception as e:
                logger.warning(f"Error in sentence transformer similarity: {e}")
                # Fallback to difflib
                similarity = difflib.SequenceMatcher(None, skill1_norm, skill2_norm).ratio()
        else:
            # Use difflib for string similarity as fallback
            similarity = difflib.SequenceMatcher(None, skill1_norm, skill2_norm).ratio()
        
        # Apply domain-specific rules
        similarity = self._apply_domain_similarity_rules(skill1_norm, skill2_norm, similarity)
        
        return similarity
    
    def _apply_domain_similarity_rules(self, skill1: str, skill2: str, 
                                      base_similarity: float) -> float:
        """
        Apply domain-specific similarity rules for better matching.
        
        Args:
            skill1: First skill (normalized)
            skill2: Second skill (normalized)
            base_similarity: Base similarity score
            
        Returns:
            Adjusted similarity score
        """
        # Programming language synonyms
        programming_synonyms = {
            'javascript': ['js', 'node.js', 'nodejs'],
            'python': ['py'],
            'c++': ['cpp', 'c plus plus'],
            'c#': ['csharp', 'c sharp'],
            'typescript': ['ts']
        }
        
        # Framework/library relationships
        framework_relationships = {
            'react': ['reactjs', 'react.js'],
            'angular': ['angularjs', 'angular.js'],
            'vue': ['vuejs', 'vue.js'],
            'express': ['expressjs', 'express.js']
        }
        
        # Cloud platform synonyms
        cloud_synonyms = {
            'aws': ['amazon web services'],
            'gcp': ['google cloud platform', 'google cloud'],
            'azure': ['microsoft azure']
        }
        
        all_synonyms = {**programming_synonyms, **framework_relationships, **cloud_synonyms}
        
        # Check for synonym matches
        for canonical, synonyms in all_synonyms.items():
            if ((skill1 == canonical and skill2 in synonyms) or 
                (skill2 == canonical and skill1 in synonyms) or
                (skill1 in synonyms and skill2 in synonyms)):
                return max(base_similarity, 0.95)
        
        # Technology stack relationships
        if self._are_related_technologies(skill1, skill2):
            return max(base_similarity, 0.7)
        
        return base_similarity
    
    def _are_related_technologies(self, skill1: str, skill2: str) -> bool:
        """
        Check if two technologies are related in the same stack.
        
        Args:
            skill1: First technology
            skill2: Second technology
            
        Returns:
            True if technologies are related
        """
        tech_stacks = [
            ['react', 'redux', 'jsx', 'javascript'],
            ['angular', 'typescript', 'rxjs'],
            ['vue', 'vuex', 'nuxt'],
            ['python', 'django', 'flask', 'fastapi'],
            ['java', 'spring', 'hibernate'],
            ['node.js', 'express', 'javascript'],
            ['aws', 'ec2', 's3', 'lambda', 'cloudformation'],
            ['docker', 'kubernetes', 'containerization'],
            ['sql', 'mysql', 'postgresql', 'database']
        ]
        
        for stack in tech_stacks:
            if skill1 in stack and skill2 in stack:
                return True
        
        return False
    
    def _calculate_batch_skill_similarity(self, skills1: List[str], skills2: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between two lists of skills using batch processing.
        
        Args:
            skills1: First list of skills
            skills2: Second list of skills
            
        Returns:
            Similarity matrix where element [i,j] is similarity between skills1[i] and skills2[j]
        """
        if self.sentence_model is not None and len(skills1) > 0 and len(skills2) > 0:
            try:
                # Normalize skills
                skills1_norm = [skill.lower().strip() for skill in skills1]
                skills2_norm = [skill.lower().strip() for skill in skills2]
                
                # Get embeddings for all skills at once
                embeddings1 = self.sentence_model.encode(skills1_norm)
                embeddings2 = self.sentence_model.encode(skills2_norm)
                
                # Calculate cosine similarity matrix
                similarity_matrix = np.dot(embeddings1, embeddings2.T) / (
                    np.linalg.norm(embeddings1, axis=1, keepdims=True) * 
                    np.linalg.norm(embeddings2, axis=1, keepdims=True).T
                )
                
                # Ensure all values are between 0 and 1
                similarity_matrix = np.clip(similarity_matrix, 0, 1)
                
                return similarity_matrix
            except Exception as e:
                logger.warning(f"Error in batch similarity calculation: {e}")
        
        # Fallback to individual calculations
        similarity_matrix = np.zeros((len(skills1), len(skills2)))
        for i, skill1 in enumerate(skills1):
            for j, skill2 in enumerate(skills2):
                similarity_matrix[i, j] = self._calculate_skill_similarity(skill1, skill2)
        
        return similarity_matrix
    
    def _is_skill_covered(self, target_skill: str, skill_list: List[str]) -> bool:
        """
        Check if a target skill is covered by any skill in the list.
        
        Args:
            target_skill: Skill to check for
            skill_list: List of skills to check against
            
        Returns:
            True if skill is covered
        """
        for skill in skill_list:
            if self._calculate_skill_similarity(target_skill, skill) >= \
               self.similarity_thresholds['moderate_similarity']:
                return True
        return False
    
    def _analyze_experience_match(self, resume_experience: ExperienceProfile, 
                                 job_requirements: JobRequirements) -> Dict[str, Any]:
        """
        Analyze experience alignment between resume and job requirements.
        
        Args:
            resume_experience: Experience profile from resume
            job_requirements: Job requirements
            
        Returns:
            Dictionary with experience analysis
        """
        required_years = job_requirements.experience_required
        candidate_years = resume_experience.years_experience
        
        # Calculate experience gap
        gap_years = candidate_years - required_years
        
        # Determine alignment level
        if gap_years >= 2:
            alignment = "perfect"
        elif gap_years >= 0:
            alignment = "good"
        elif gap_years >= -1:
            alignment = "acceptable"
        else:
            alignment = "insufficient"
        
        # Find relevant experience areas
        relevant_areas = []
        for specialization in resume_experience.specializations:
            if any(self._calculate_skill_similarity(specialization, req) > 0.6 
                  for req in job_requirements.required_skills + job_requirements.preferred_skills):
                relevant_areas.append(specialization)
        
        return {
            'required_years': required_years,
            'candidate_years': candidate_years,
            'gap_years': gap_years,
            'alignment': alignment,
            'relevant_areas': relevant_areas,
            'experience_level_match': self._compare_experience_levels(
                resume_experience.experience_level, job_requirements.job_level
            )
        }
    
    def _compare_experience_levels(self, candidate_level: str, required_level: str) -> str:
        """
        Compare candidate and required experience levels.
        
        Args:
            candidate_level: Candidate's experience level
            required_level: Required experience level
            
        Returns:
            Comparison result
        """
        level_hierarchy = ['entry', 'junior', 'mid', 'senior', 'lead', 'principal']
        
        try:
            candidate_idx = level_hierarchy.index(candidate_level.lower())
            required_idx = level_hierarchy.index(required_level.lower())
            
            if candidate_idx >= required_idx:
                return "meets_or_exceeds"
            elif candidate_idx == required_idx - 1:
                return "close_match"
            else:
                return "below_requirement"
        except ValueError:
            return "unknown"
    
    def _analyze_education_match(self, resume_text: str, 
                                job_requirements: JobRequirements) -> Dict[str, Any]:
        """
        Analyze education requirements matching.
        
        Args:
            resume_text: Full resume text
            job_requirements: Job requirements
            
        Returns:
            Dictionary with education analysis
        """
        # Extract education information from resume
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 
            'college', 'education', 'graduated', 'diploma', 'certification'
        ]
        
        resume_lower = resume_text.lower()
        has_education = any(keyword in resume_lower for keyword in education_keywords)
        
        # Determine education level
        education_level = "unknown"
        if 'phd' in resume_lower or 'doctorate' in resume_lower:
            education_level = "doctorate"
        elif 'master' in resume_lower:
            education_level = "masters"
        elif 'bachelor' in resume_lower:
            education_level = "bachelors"
        elif 'associate' in resume_lower:
            education_level = "associates"
        
        # Check if meets requirements
        required_education = job_requirements.education_required.lower()
        meets_requirements = True
        
        if 'bachelor' in required_education and education_level not in ['bachelors', 'masters', 'doctorate']:
            meets_requirements = False
        elif 'master' in required_education and education_level not in ['masters', 'doctorate']:
            meets_requirements = False
        
        return {
            'has_education_info': has_education,
            'education_level': education_level,
            'required_education': required_education,
            'meets_requirements': meets_requirements,
            'level_comparison': self._compare_education_levels(education_level, required_education)
        }
    
    def _compare_education_levels(self, candidate_level: str, required_level: str) -> str:
        """
        Compare education levels.
        
        Args:
            candidate_level: Candidate's education level
            required_level: Required education level
            
        Returns:
            Comparison result
        """
        level_hierarchy = ['associates', 'bachelors', 'masters', 'doctorate']
        
        try:
            if candidate_level == "unknown":
                return "unknown"
            
            candidate_idx = level_hierarchy.index(candidate_level)
            
            if 'bachelor' in required_level:
                required_idx = level_hierarchy.index('bachelors')
            elif 'master' in required_level:
                required_idx = level_hierarchy.index('masters')
            elif 'doctorate' in required_level or 'phd' in required_level:
                required_idx = level_hierarchy.index('doctorate')
            else:
                return "meets_requirement"
            
            if candidate_idx >= required_idx:
                return "exceeds_requirement"
            else:
                return "below_requirement"
                
        except (ValueError, IndexError):
            return "unknown"
    
    def _calculate_skills_score(self, skill_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall skills matching score using enhanced semantic similarity.
        
        Args:
            skill_analysis: Skill analysis results
            
        Returns:
            Skills score (0-100)
        """
        total_required = skill_analysis['total_required_skills']
        
        if total_required == 0:
            return 100.0
        
        # Use coverage percentage from detailed matching
        required_coverage = skill_analysis.get('required_skills_coverage', 0.0)
        preferred_coverage = skill_analysis.get('preferred_skills_coverage', 0.0)
        
        # Base score from required skills coverage (weighted by confidence)
        confidence_weight = skill_analysis.get('confidence_scores', {}).get('required_skills', 1.0)
        base_score = required_coverage * 80 * confidence_weight  # Max 80 points for required skills
        
        # Bonus points for preferred skills
        preferred_bonus = preferred_coverage * 15  # Max 15 points for preferred skills
        
        # Bonus for additional skills
        additional_bonus = min(len(skill_analysis['additional_skills']) * 1, 5)
        
        total_score = base_score + preferred_bonus + additional_bonus
        return min(total_score, 100.0)
    
    def _calculate_experience_score(self, experience_analysis: Dict[str, Any]) -> float:
        """
        Calculate experience matching score.
        
        Args:
            experience_analysis: Experience analysis results
            
        Returns:
            Experience score (0-100)
        """
        alignment = experience_analysis['alignment']
        
        alignment_scores = {
            'perfect': 100,
            'good': 85,
            'acceptable': 70,
            'insufficient': 40
        }
        
        base_score = alignment_scores.get(alignment, 50)
        
        # Bonus for relevant experience areas
        relevant_bonus = min(len(experience_analysis['relevant_areas']) * 5, 15)
        
        return min(base_score + relevant_bonus, 100.0)
    
    def _calculate_education_score(self, education_analysis: Dict[str, Any]) -> float:
        """
        Calculate education matching score.
        
        Args:
            education_analysis: Education analysis results
            
        Returns:
            Education score (0-100)
        """
        if education_analysis['meets_requirements']:
            return 100.0
        elif education_analysis['level_comparison'] == 'unknown':
            return 70.0  # Neutral score if education info is unclear
        else:
            return 50.0  # Below requirements
    
    def _calculate_confidence_score(self, resume_skills: ExtractedSkills, 
                                   resume_experience: ExperienceProfile,
                                   job_requirements: JobRequirements) -> float:
        """
        Calculate confidence score for the matching analysis.
        
        Args:
            resume_skills: Extracted skills from resume
            resume_experience: Experience profile from resume
            job_requirements: Job requirements
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from extraction quality
        skills_confidence = resume_skills.confidence_score
        experience_confidence = resume_experience.confidence_score
        
        # Adjust based on data completeness
        completeness_factors = [
            len(resume_skills.technical_skills) > 0,
            len(resume_skills.soft_skills) > 0,
            resume_experience.years_experience > 0,
            len(job_requirements.required_skills) > 0,
            len(job_requirements.preferred_skills) > 0
        ]
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Combine scores
        overall_confidence = (
            skills_confidence * 0.4 +
            experience_confidence * 0.3 +
            completeness_score * 0.3
        )
        
        return round(overall_confidence, 3)
    
    def _calculate_enhanced_confidence_score(self, skills_score: float, 
                                           experience_score: float, 
                                           education_score: float,
                                           confidence_scores: Dict[str, float]) -> float:
        """
        Calculate enhanced confidence score incorporating semantic similarity confidence.
        
        Args:
            skills_score: Skills matching score
            experience_score: Experience matching score
            education_score: Education matching score
            confidence_scores: Confidence scores from semantic matching
            
        Returns:
            Enhanced confidence score (0-100)
        """
        # Base confidence from component scores
        avg_score = (skills_score + experience_score + education_score) / 3
        
        # Incorporate semantic similarity confidence
        semantic_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.7
        
        # Adjust based on score distribution
        score_variance = np.var([skills_score, experience_score, education_score])
        variance_penalty = min(score_variance / 100, 0.2)  # Max 20% penalty
        
        # Combine traditional confidence with semantic confidence
        base_confidence = max(0, min(100, avg_score - variance_penalty * 100))
        enhanced_confidence = base_confidence * (0.7 + 0.3 * semantic_confidence)
        
        return min(100, enhanced_confidence)
    
    def _generate_recommendations(self, skill_analysis: Dict[str, Any],
                                 experience_analysis: Dict[str, Any],
                                 education_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations based on matching analysis.
        
        Args:
            skill_analysis: Skill analysis results
            experience_analysis: Experience analysis results
            education_analysis: Education analysis results
            
        Returns:
            Dictionary with different types of recommendations
        """
        improvements = []
        strengths = []
        talking_points = []
        
        # Skill-based recommendations
        if skill_analysis['missing_required']:
            improvements.append(
                f"Develop skills in: {', '.join(skill_analysis['missing_required'][:3])}"
            )
        
        if skill_analysis['matched_technical']:
            strengths.append(
                f"Strong technical skills: {', '.join(skill_analysis['matched_technical'][:3])}"
            )
            talking_points.append(
                f"Highlight experience with {', '.join(skill_analysis['matched_technical'][:2])}"
            )
        
        # Enhanced recommendations based on confidence scores
        detailed_matches = skill_analysis.get('detailed_required_matches', [])
        low_confidence_matches = [
            match for match in detailed_matches 
            if match.get('confidence') in ['low', 'very_low']
        ]
        
        if low_confidence_matches:
            skills_to_strengthen = [match['job_skill'] for match in low_confidence_matches[:2]]
            improvements.append(
                f"Strengthen expertise in: {', '.join(skills_to_strengthen)} to better match requirements"
            )
        
        # Experience-based recommendations
        if experience_analysis['alignment'] == 'insufficient':
            gap = abs(experience_analysis['gap_years'])
            improvements.append(
                f"Gain {gap} more years of relevant experience"
            )
        elif experience_analysis['alignment'] in ['good', 'perfect']:
            strengths.append(
                f"Excellent experience level ({experience_analysis['candidate_years']} years)"
            )
        
        if experience_analysis['relevant_areas']:
            talking_points.append(
                f"Emphasize experience in {', '.join(experience_analysis['relevant_areas'][:2])}"
            )
        
        # Education-based recommendations
        if not education_analysis['meets_requirements']:
            improvements.append(
                f"Consider pursuing {education_analysis['required_education']}"
            )
        
        # Additional skills as strengths
        if skill_analysis['additional_skills']:
            strengths.append(
                f"Additional valuable skills: {', '.join(skill_analysis['additional_skills'][:3])}"
            )
        
        # Coverage-based recommendations
        required_coverage = skill_analysis.get('required_skills_coverage', 0)
        if required_coverage < 0.7:
            improvements.append(
                "Focus on building core technical skills to meet job requirements"
            )
        
        return {
            'improvements': improvements,
            'strengths': strengths,
            'talking_points': talking_points
        }
    
    def _generate_enhanced_recommendations(self, skill_analysis: Dict[str, Any],
                                         experience_analysis: Dict[str, Any],
                                         education_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate enhanced personalized recommendations using semantic similarity insights.
        
        Args:
            skill_analysis: Enhanced skills matching analysis with confidence scores
            experience_analysis: Experience matching analysis
            education_analysis: Education matching analysis
            
        Returns:
            List of detailed recommendation strings
        """
        recommendations = []
        
        # Enhanced skills recommendations with confidence consideration
        if skill_analysis['missing_required']:
            missing_skills = skill_analysis['missing_required'][:3]
            recommendations.append(
                f"Priority skills to develop: {', '.join(missing_skills)}"
            )
        
        # Low confidence matches - suggest improvement
        detailed_matches = skill_analysis.get('detailed_required_matches', [])
        low_confidence_matches = [
            match for match in detailed_matches 
            if match.get('confidence') in ['low', 'very_low']
        ]
        
        if low_confidence_matches:
            skills_to_strengthen = [match['job_skill'] for match in low_confidence_matches[:2]]
            recommendations.append(
                f"Strengthen your expertise in: {', '.join(skills_to_strengthen)} to better match job requirements"
            )
        
        # Preferred skills with high potential impact
        if skill_analysis['missing_preferred']:
            preferred_skills = skill_analysis['missing_preferred'][:2]
            recommendations.append(
                f"High-impact preferred skills: {', '.join(preferred_skills)}"
            )
        
        # Experience recommendations
        if experience_analysis.get('years_gap', 0) > 0:
            recommendations.append(
                f"Target {experience_analysis['years_gap']} more years of relevant experience"
            )
        
        if experience_analysis.get('missing_areas'):
            missing_areas = experience_analysis['missing_areas'][:2]
            recommendations.append(
                f"Expand experience in: {', '.join(missing_areas)}"
            )
        
        # Education recommendations
        if not education_analysis.get('degree_match', True):
            recommendations.append(
                "Consider relevant certifications or degree programs"
            )
        
        # Coverage-based recommendations
        required_coverage = skill_analysis.get('required_skills_coverage', 0)
        if required_coverage < 0.7:
            recommendations.append(
                "Focus on building core technical skills to meet job requirements"
            )
        
        return recommendations
    
    def _generate_match_explanation(self, overall_score: float, 
                                   skill_analysis: Dict[str, Any],
                                   experience_analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the match score.
        
        Args:
            overall_score: Overall matching score
            skill_analysis: Skill analysis results
            experience_analysis: Experience analysis results
            
        Returns:
            Match explanation string
        """
        if overall_score >= 90:
            level = "Excellent"
        elif overall_score >= 80:
            level = "Very Good"
        elif overall_score >= 70:
            level = "Good"
        elif overall_score >= 60:
            level = "Fair"
        else:
            level = "Poor"
        
        matched_skills = skill_analysis['total_matched_skills']
        required_skills = skill_analysis['total_required_skills']
        experience_alignment = experience_analysis['alignment']
        
        explanation = (
            f"{level} match ({overall_score:.1f}%). "
            f"Candidate matches {matched_skills}/{required_skills} required skills "
            f"with {experience_alignment} experience alignment."
        )
        
        return explanation
    
    def _calculate_average_confidence(self, matched_skills: List[Dict[str, Any]]) -> float:
        """
        Calculate average confidence score for matched skills.
        
        Args:
            matched_skills: List of matched skills with confidence scores
            
        Returns:
            Average confidence score (0-1)
        """
        if not matched_skills:
            return 0.0
        
        confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5, 'very_low': 0.3}
        total_confidence = sum(
            confidence_map.get(match.get('confidence', 'low'), 0.5) 
            for match in matched_skills
        )
        
        return total_confidence / len(matched_skills)
    
    def export_match_report(self, match_result: DetailedMatchResult, 
                           output_path: str = None) -> str:
        """
        Export detailed match report to JSON file.
        
        Args:
            match_result: Detailed match result
            output_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"match_report_{timestamp}.json"
        
        # Convert to dictionary
        report_data = asdict(match_result)
        
        # Add metadata
        report_data['export_timestamp'] = datetime.now().isoformat()
        report_data['report_version'] = '1.0'
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Match report exported to: {output_path}")
        return output_path


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe
    Senior Software Engineer
    
    EXPERIENCE:
    5 years of professional software development experience.
    
    TECHNICAL SKILLS:
    Python, JavaScript, React, Node.js, AWS, Docker, SQL, Git
    
    WORK EXPERIENCE:
    Senior Software Engineer - Tech Corp (3 years)
    - Developed scalable web applications using React and Node.js
    - Implemented microservices architecture with Docker and AWS
    - Led a team of 3 junior developers
    
    Software Engineer - StartupXYZ (2 years)
    - Built REST APIs using Python and Flask
    - Managed PostgreSQL databases
    - Collaborated with cross-functional teams
    
    EDUCATION:
    Bachelor of Science in Computer Science
    University of Technology, 2018
    """
    
    sample_job = """
    Senior Full Stack Developer
    
    We are looking for an experienced Senior Full Stack Developer to join our team.
    
    REQUIRED SKILLS:
    - 4+ years of software development experience
    - Proficiency in JavaScript, React, Node.js
    - Experience with cloud platforms (AWS preferred)
    - Database management (SQL)
    - Version control (Git)
    
    PREFERRED SKILLS:
    - Python programming
    - Docker containerization
    - Agile development methodologies
    - Team leadership experience
    
    REQUIREMENTS:
    - Bachelor's degree in Computer Science or related field
    - Strong problem-solving skills
    - Excellent communication abilities
    """
    
    # Initialize matcher
    matcher = AdvancedJobMatcher()
    
    # Perform matching
    try:
        result = matcher.match_resume_to_job(sample_resume, sample_job)
        
        print("\n" + "="*60)
        print("JOB MATCHING ANALYSIS RESULTS")
        print("="*60)
        print(f"Overall Score: {result.overall_score}%")
        print(f"Skills Score: {result.skills_score}%")
        print(f"Experience Score: {result.experience_score}%")
        print(f"Education Score: {result.education_score}%")
        print(f"\nConfidence: {result.confidence_score:.2f}")
        print(f"\nExplanation: {result.match_explanation}")
        
        print("\nMatched Skills:")
        for skill in result.matched_technical_skills:
            print(f"  ✓ {skill}")
        
        print("\nMissing Required Skills:")
        for skill in result.missing_required_skills:
            print(f"  ✗ {skill}")
        
        print("\nRecommendations:")
        for rec in result.improvement_recommendations:
            print(f"  • {rec}")
        
        # Export report
        report_path = matcher.export_match_report(result)
        print(f"\nDetailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error in job matching: {e}")