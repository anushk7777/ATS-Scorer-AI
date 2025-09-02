#!/usr/bin/env python3
"""
Enhanced Salary Predictor with Gemini Integration

This module combines the existing ML salary prediction model with Gemini's semantic
keyword extraction capabilities to provide more accurate salary predictions based on
semanticially extracted features from resumes.

Features:
- Integration with Gemini Semantic Extractor
- Enhanced feature engineering using semantic data
- Improved ML model training with semantic features
- RESTful API endpoints for web integration
- Comprehensive logging and error handling

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our custom modules
try:
    from gemini_semantic_extractor import (
        GeminiSemanticExtractor, ExtractedSkills, ExperienceProfile,
        JobRequirements, SkillsMatchResult
    )
except ImportError:
    print("Warning: gemini_semantic_extractor not found")
    GeminiSemanticExtractor = None

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedFeatures:
    """
    Data class for enhanced features extracted using semantic analysis
    """
    # Basic features
    years_of_experience: float
    experience_level: str
    location: str
    education_level: int
    
    # Semantic features from Gemini
    technical_skills_count: int
    programming_languages_count: int
    frameworks_tools_count: int
    certifications_count: int
    domain_expertise_count: int
    soft_skills_count: int
    
    # Advanced semantic features
    skill_rarity_score: float
    skill_market_value_score: float
    leadership_indicator: float
    project_complexity_score: float
    industry_relevance_score: float
    
    # Confidence metrics
    extraction_confidence: float
    feature_quality_score: float

@dataclass
class SalaryPredictionResult:
    """
    Data class for salary prediction results
    """
    predicted_salary: float
    confidence_score: float
    salary_range: Tuple[float, float]
    key_factors: List[str]
    market_comparison: str
    feature_importance: Dict[str, float]
    semantic_analysis: Dict[str, Any]
    prediction_timestamp: str

class EnhancedSalaryPredictor:
    """
    Enhanced Salary Predictor that combines ML models with semantic analysis
    
    This class provides comprehensive salary prediction capabilities by leveraging
    both traditional ML features and semantic features extracted using Gemini API.
    """
    
    def __init__(self, model_path: Optional[str] = None, gemini_api_key: Optional[str] = None):
        """
        Initialize the Enhanced Salary Predictor
        
        Args:
            model_path: Path to pre-trained model file
            gemini_api_key: Gemini API key for semantic extraction
        """
        self.model_path = model_path or 'enhanced_salary_model.joblib'
        self.semantic_extractor = None
        self.ml_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_importance = {}
        
        # Initialize semantic extractor
        if GeminiSemanticExtractor:
            self.semantic_extractor = GeminiSemanticExtractor(gemini_api_key)
        
        # Skill value mappings for market analysis
        self.skill_market_values = {
            # High-value skills
            'machine learning': 1.0, 'artificial intelligence': 1.0, 'deep learning': 0.95,
            'data science': 0.9, 'cloud computing': 0.85, 'kubernetes': 0.85,
            'aws': 0.8, 'azure': 0.8, 'gcp': 0.8, 'docker': 0.75,
            'react': 0.7, 'angular': 0.7, 'vue': 0.65, 'node.js': 0.7,
            'python': 0.75, 'java': 0.7, 'javascript': 0.65, 'typescript': 0.7,
            'go': 0.8, 'rust': 0.85, 'scala': 0.8,
            
            # Medium-value skills
            'sql': 0.6, 'postgresql': 0.65, 'mongodb': 0.6,
            'django': 0.65, 'flask': 0.6, 'spring': 0.65,
            'git': 0.5, 'jenkins': 0.6, 'ci/cd': 0.65,
            
            # Standard skills
            'html': 0.4, 'css': 0.4, 'php': 0.5,
            'mysql': 0.5, 'linux': 0.55
        }
        
        # Industry multipliers
        self.industry_multipliers = {
            'technology': 1.2, 'finance': 1.15, 'healthcare': 1.1,
            'consulting': 1.1, 'e-commerce': 1.05, 'gaming': 1.0,
            'education': 0.9, 'non-profit': 0.85
        }
        
        # Load existing model if available
        self._load_model()
    
    def extract_enhanced_features(self, resume_text: str, location: str = 'bangalore') -> EnhancedFeatures:
        """
        Extract enhanced features from resume text using semantic analysis
        
        Args:
            resume_text: The resume text to analyze
            location: Candidate location
            
        Returns:
            EnhancedFeatures object containing all extracted features
        """
        if not self.semantic_extractor:
            logger.warning("Semantic extractor not available, using fallback")
            return self._fallback_feature_extraction(resume_text, location)
        
        try:
            # Extract semantic information
            skills = self.semantic_extractor.extract_skills_from_resume(resume_text)
            experience = self.semantic_extractor.extract_experience_profile(resume_text)
            
            # Calculate basic counts
            technical_skills_count = len(skills.technical_skills)
            programming_languages_count = len(skills.programming_languages)
            frameworks_tools_count = len(skills.frameworks_tools)
            certifications_count = len(skills.certifications)
            domain_expertise_count = len(skills.domains)
            soft_skills_count = len(skills.soft_skills)
            
            # Calculate advanced semantic features
            skill_rarity_score = self._calculate_skill_rarity_score(skills)
            skill_market_value_score = self._calculate_skill_market_value(skills)
            leadership_indicator = 1.0 if experience.leadership_experience else 0.0
            project_complexity_score = self._map_complexity_to_score(experience.project_complexity)
            industry_relevance_score = self._calculate_industry_relevance(skills, experience)
            
            # Education level mapping
            education_level = self._extract_education_level(resume_text)
            
            # Calculate confidence metrics
            extraction_confidence = (skills.confidence_score + experience.confidence_score) / 2
            feature_quality_score = self._calculate_feature_quality(
                technical_skills_count, programming_languages_count, 
                frameworks_tools_count, experience.years_of_experience
            )
            
            return EnhancedFeatures(
                years_of_experience=experience.years_of_experience,
                experience_level=experience.experience_level,
                location=location.lower(),
                education_level=education_level,
                technical_skills_count=technical_skills_count,
                programming_languages_count=programming_languages_count,
                frameworks_tools_count=frameworks_tools_count,
                certifications_count=certifications_count,
                domain_expertise_count=domain_expertise_count,
                soft_skills_count=soft_skills_count,
                skill_rarity_score=skill_rarity_score,
                skill_market_value_score=skill_market_value_score,
                leadership_indicator=leadership_indicator,
                project_complexity_score=project_complexity_score,
                industry_relevance_score=industry_relevance_score,
                extraction_confidence=extraction_confidence,
                feature_quality_score=feature_quality_score
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced feature extraction: {e}")
            return self._fallback_feature_extraction(resume_text, location)
    
    def predict_salary(self, resume_text: str, location: str = 'bangalore') -> SalaryPredictionResult:
        """
        Predict salary using enhanced features and semantic analysis
        
        Args:
            resume_text: The resume text to analyze
            location: Candidate location
            
        Returns:
            SalaryPredictionResult object containing prediction and analysis
        """
        # Extract enhanced features
        features = self.extract_enhanced_features(resume_text, location)
        
        # Convert features to model input
        feature_vector = self._features_to_vector(features)
        
        # Make prediction
        if self.ml_model and self.scaler:
            try:
                # Scale features
                scaled_features = self.scaler.transform([feature_vector])
                
                # Predict salary
                predicted_salary = self.ml_model.predict(scaled_features)[0]
                
                # Calculate confidence based on feature quality and model confidence
                confidence_score = min(features.feature_quality_score * features.extraction_confidence, 0.95)
                
                # Calculate salary range (±15% based on confidence)
                range_factor = 0.15 * (1 - confidence_score) + 0.05
                salary_range = (
                    predicted_salary * (1 - range_factor),
                    predicted_salary * (1 + range_factor)
                )
                
                # Generate key factors
                key_factors = self._generate_key_factors(features)
                
                # Market comparison
                market_comparison = self._generate_market_comparison(predicted_salary, features)
                
                # Feature importance (if available)
                feature_importance = self._get_feature_importance()
                
                # Semantic analysis summary
                semantic_analysis = {
                    'skills_extracted': features.technical_skills_count + features.programming_languages_count,
                    'experience_level': features.experience_level,
                    'market_value_score': features.skill_market_value_score,
                    'leadership_indicator': features.leadership_indicator,
                    'extraction_confidence': features.extraction_confidence
                }
                
                return SalaryPredictionResult(
                    predicted_salary=round(predicted_salary, 2),
                    confidence_score=round(confidence_score, 2),
                    salary_range=(round(salary_range[0], 2), round(salary_range[1], 2)),
                    key_factors=key_factors,
                    market_comparison=market_comparison,
                    feature_importance=feature_importance,
                    semantic_analysis=semantic_analysis,
                    prediction_timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
        
        # Fallback prediction
        return self._fallback_prediction(features)
    
    def match_job_description(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Match resume with job description and provide detailed analysis
        
        Args:
            resume_text: The resume text
            job_description: The job description text
            
        Returns:
            Dictionary containing matching analysis
        """
        if not self.semantic_extractor:
            return {'error': 'Semantic extractor not available'}
        
        try:
            # Extract features from both resume and job description
            resume_skills = self.semantic_extractor.extract_skills_from_resume(resume_text)
            resume_experience = self.semantic_extractor.extract_experience_profile(resume_text)
            job_requirements = self.semantic_extractor.extract_job_requirements(job_description)
            
            # Perform skills matching
            match_result = self.semantic_extractor.match_skills(resume_skills, job_requirements)
            
            # Calculate salary prediction for this specific role
            salary_prediction = self.predict_salary(resume_text)
            
            # Generate comprehensive analysis
            analysis = {
                'overall_match_score': round(match_result.overall_match_score * 100, 1),
                'skills_analysis': {
                    'matched_skills': match_result.matched_skills,
                    'missing_skills': match_result.missing_skills,
                    'additional_skills': match_result.additional_skills,
                    'detailed_breakdown': {
                        k: round(v * 100, 1) for k, v in match_result.detailed_breakdown.items()
                    }
                },
                'experience_analysis': {
                    'candidate_experience': resume_experience.years_of_experience,
                    'required_experience': job_requirements.experience_required,
                    'experience_match': self._evaluate_experience_match(
                        resume_experience.years_of_experience, 
                        job_requirements.experience_required
                    ),
                    'level_compatibility': resume_experience.experience_level
                },
                'salary_prediction': asdict(salary_prediction),
                'recommendations': self._generate_recommendations(match_result, resume_experience, job_requirements),
                'fit_assessment': self._assess_overall_fit(match_result, resume_experience, job_requirements)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in job matching: {e}")
            return {'error': f'Job matching failed: {str(e)}'}
    
    def train_enhanced_model(self, training_data_path: str) -> Dict[str, Any]:
        """
        Train the enhanced ML model using semantic features
        
        Args:
            training_data_path: Path to training data CSV file
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Load training data
            df = pd.read_csv(training_data_path)
            
            # Extract features for each resume in training data
            enhanced_features_list = []
            for _, row in df.iterrows():
                if 'resume_text' in row and pd.notna(row['resume_text']):
                    features = self.extract_enhanced_features(
                        str(row['resume_text']), 
                        row.get('location', 'bangalore')
                    )
                    enhanced_features_list.append(self._features_to_vector(features))
                else:
                    # Use fallback features if resume text not available
                    enhanced_features_list.append(self._create_fallback_vector(row))
            
            # Prepare training data
            X = np.array(enhanced_features_list)
            y = df['salary'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.ml_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store feature importance
            feature_names = self._get_feature_names()
            self.feature_importance = dict(zip(
                feature_names, 
                self.ml_model.feature_importances_
            ))
            
            # Save model
            self._save_model()
            
            training_results = {
                'model_performance': {
                    'mean_absolute_error': round(mae, 2),
                    'r2_score': round(r2, 3),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'feature_importance': {
                    k: round(v, 3) for k, v in sorted(
                        self.feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                },
                'training_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model training completed. MAE: {mae:.2f}, R2: {r2:.3f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'error': f'Training failed: {str(e)}'}
    
    def _calculate_skill_rarity_score(self, skills: ExtractedSkills) -> float:
        """
        Calculate rarity score based on skill combinations
        """
        all_skills = (
            skills.technical_skills + skills.programming_languages + 
            skills.frameworks_tools + skills.domains
        )
        
        # High-value/rare skills
        rare_skills = [
            'machine learning', 'artificial intelligence', 'deep learning',
            'kubernetes', 'go', 'rust', 'scala', 'blockchain'
        ]
        
        rare_count = sum(1 for skill in all_skills if any(rare in skill.lower() for rare in rare_skills))
        total_skills = len(all_skills)
        
        if total_skills == 0:
            return 0.0
        
        return min(rare_count / total_skills * 2, 1.0)  # Cap at 1.0
    
    def _calculate_skill_market_value(self, skills: ExtractedSkills) -> float:
        """
        Calculate market value score based on skill demand
        """
        all_skills = (
            skills.technical_skills + skills.programming_languages + 
            skills.frameworks_tools
        )
        
        total_value = 0.0
        skill_count = 0
        
        for skill in all_skills:
            skill_lower = skill.lower()
            for market_skill, value in self.skill_market_values.items():
                if market_skill in skill_lower or skill_lower in market_skill:
                    total_value += value
                    skill_count += 1
                    break
        
        return total_value / max(skill_count, 1)
    
    def _map_complexity_to_score(self, complexity: str) -> float:
        """
        Map project complexity to numerical score
        """
        complexity_map = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
        return complexity_map.get(complexity.lower(), 0.5)
    
    def _calculate_industry_relevance(self, skills: ExtractedSkills, experience: ExperienceProfile) -> float:
        """
        Calculate industry relevance score
        """
        # Simple implementation - can be enhanced
        tech_domains = ['technology', 'software', 'it', 'computer']
        
        domain_match = any(
            any(tech in domain.lower() for tech in tech_domains)
            for domain in skills.domains
        )
        
        industry_match = any(
            any(tech in industry.lower() for tech in tech_domains)
            for industry in experience.industries
        )
        
        return 1.0 if (domain_match or industry_match) else 0.5
    
    def _extract_education_level(self, resume_text: str) -> int:
        """
        Extract education level from resume text
        """
        text_lower = resume_text.lower()
        
        if any(degree in text_lower for degree in ['phd', 'ph.d', 'doctorate']):
            return 4
        elif any(degree in text_lower for degree in ['master', 'mba', 'ms', 'm.tech', 'm.s']):
            return 3
        elif any(degree in text_lower for degree in ['bachelor', 'b.tech', 'b.e', 'b.s', 'btech']):
            return 2
        elif any(degree in text_lower for degree in ['diploma', 'associate']):
            return 1
        else:
            return 2  # Default to bachelor's
    
    def _calculate_feature_quality(self, tech_count: int, prog_count: int, 
                                 tools_count: int, experience: float) -> float:
        """
        Calculate overall feature quality score
        """
        # Normalize counts
        tech_score = min(tech_count / 10, 1.0)
        prog_score = min(prog_count / 5, 1.0)
        tools_score = min(tools_count / 8, 1.0)
        exp_score = min(experience / 10, 1.0)
        
        return (tech_score + prog_score + tools_score + exp_score) / 4
    
    def _features_to_vector(self, features: EnhancedFeatures) -> List[float]:
        """
        Convert EnhancedFeatures to numerical vector for ML model
        """
        # Encode categorical features
        experience_level_map = {'fresher': 0, 'junior': 1, 'mid_level': 2, 'senior': 3}
        location_map = {'bangalore': 0, 'mumbai': 1, 'delhi': 2, 'hyderabad': 3, 'pune': 4, 'chennai': 5}
        
        return [
            features.years_of_experience,
            experience_level_map.get(features.experience_level, 0),
            location_map.get(features.location, 0),
            features.education_level,
            features.technical_skills_count,
            features.programming_languages_count,
            features.frameworks_tools_count,
            features.certifications_count,
            features.domain_expertise_count,
            features.soft_skills_count,
            features.skill_rarity_score,
            features.skill_market_value_score,
            features.leadership_indicator,
            features.project_complexity_score,
            features.industry_relevance_score,
            features.extraction_confidence,
            features.feature_quality_score
        ]
    
    def _get_feature_names(self) -> List[str]:
        """
        Get feature names for model interpretation
        """
        return [
            'years_of_experience', 'experience_level', 'location', 'education_level',
            'technical_skills_count', 'programming_languages_count', 'frameworks_tools_count',
            'certifications_count', 'domain_expertise_count', 'soft_skills_count',
            'skill_rarity_score', 'skill_market_value_score', 'leadership_indicator',
            'project_complexity_score', 'industry_relevance_score', 'extraction_confidence',
            'feature_quality_score'
        ]
    
    def _generate_key_factors(self, features: EnhancedFeatures) -> List[str]:
        """
        Generate key factors affecting salary prediction
        """
        factors = []
        
        if features.years_of_experience >= 5:
            factors.append(f"Extensive experience ({features.years_of_experience} years)")
        elif features.years_of_experience >= 2:
            factors.append(f"Solid experience ({features.years_of_experience} years)")
        
        if features.skill_market_value_score > 0.7:
            factors.append("High-value technical skills")
        
        if features.leadership_indicator > 0.5:
            factors.append("Leadership experience")
        
        if features.certifications_count > 0:
            factors.append(f"Professional certifications ({features.certifications_count})")
        
        if features.education_level >= 3:
            factors.append("Advanced education (Master's or higher)")
        
        return factors[:5]  # Return top 5 factors
    
    def _generate_market_comparison(self, predicted_salary: float, features: EnhancedFeatures) -> str:
        """
        Generate market comparison text
        """
        # Simple market comparison logic
        if predicted_salary > 20:
            return "Above market average for similar profiles"
        elif predicted_salary > 15:
            return "Competitive with market standards"
        elif predicted_salary > 10:
            return "Aligned with market entry-level positions"
        else:
            return "Below market average - consider skill development"
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model
        """
        if self.feature_importance:
            return {k: round(v, 3) for k, v in sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]}
        return {}
    
    def _fallback_feature_extraction(self, resume_text: str, location: str) -> EnhancedFeatures:
        """
        Fallback feature extraction when Gemini is not available
        """
        # Simple regex-based extraction
        text_lower = resume_text.lower()
        
        # Count basic skills
        programming_languages = ['python', 'java', 'javascript', 'c++', 'c#']
        frameworks = ['react', 'angular', 'django', 'spring', 'flask']
        
        prog_count = sum(1 for lang in programming_languages if lang in text_lower)
        framework_count = sum(1 for fw in frameworks if fw in text_lower)
        
        # Extract years of experience
        import re
        exp_match = re.search(r'(\d+)\+?\s*years?\s*of\s*experience', text_lower)
        years_exp = float(exp_match.group(1)) if exp_match else 1.0
        
        # Determine experience level
        if years_exp <= 1:
            exp_level = 'fresher'
        elif years_exp <= 3:
            exp_level = 'junior'
        elif years_exp <= 7:
            exp_level = 'mid_level'
        else:
            exp_level = 'senior'
        
        return EnhancedFeatures(
            years_of_experience=years_exp,
            experience_level=exp_level,
            location=location.lower(),
            education_level=2,  # Default to bachelor's
            technical_skills_count=prog_count + framework_count,
            programming_languages_count=prog_count,
            frameworks_tools_count=framework_count,
            certifications_count=0,
            domain_expertise_count=1,
            soft_skills_count=2,
            skill_rarity_score=0.3,
            skill_market_value_score=0.5,
            leadership_indicator=0.0,
            project_complexity_score=0.5,
            industry_relevance_score=0.7,
            extraction_confidence=0.4,
            feature_quality_score=0.5
        )
    
    def _fallback_prediction(self, features: EnhancedFeatures) -> SalaryPredictionResult:
        """
        Fallback prediction when ML model is not available
        """
        # Simple rule-based prediction
        base_salary = 8.0  # Base salary in lakhs
        
        # Experience multiplier
        exp_multiplier = 1 + (features.years_of_experience * 0.15)
        
        # Skills multiplier
        skills_multiplier = 1 + (features.skill_market_value_score * 0.3)
        
        # Education multiplier
        edu_multiplier = 1 + (features.education_level - 2) * 0.1
        
        predicted_salary = base_salary * exp_multiplier * skills_multiplier * edu_multiplier
        
        return SalaryPredictionResult(
            predicted_salary=round(predicted_salary, 2),
            confidence_score=0.6,
            salary_range=(round(predicted_salary * 0.85, 2), round(predicted_salary * 1.15, 2)),
            key_factors=["Experience level", "Technical skills", "Education"],
            market_comparison="Estimated based on basic factors",
            feature_importance={},
            semantic_analysis={'note': 'Fallback prediction used'},
            prediction_timestamp=datetime.now().isoformat()
        )
    
    def _create_fallback_vector(self, row: pd.Series) -> List[float]:
        """
        Create fallback feature vector from training data row
        """
        return [
            row.get('years_of_experience', 2.0),
            0,  # experience_level
            0,  # location
            row.get('education_level', 2),
            5,  # technical_skills_count
            3,  # programming_languages_count
            2,  # frameworks_tools_count
            0,  # certifications_count
            1,  # domain_expertise_count
            2,  # soft_skills_count
            0.3,  # skill_rarity_score
            0.5,  # skill_market_value_score
            0.0,  # leadership_indicator
            0.5,  # project_complexity_score
            0.7,  # industry_relevance_score
            0.5,  # extraction_confidence
            0.5   # feature_quality_score
        ]
    
    def _evaluate_experience_match(self, candidate_exp: float, required_exp: str) -> bool:
        """
        Evaluate if candidate experience matches job requirements
        """
        if not required_exp:
            return True
        
        # Extract numbers from required experience string
        import re
        numbers = re.findall(r'\d+', required_exp)
        if numbers:
            min_required = int(numbers[0])
            return candidate_exp >= min_required
        
        return True
    
    def _generate_recommendations(self, match_result: SkillsMatchResult, 
                                resume_exp: ExperienceProfile, 
                                job_reqs: JobRequirements) -> List[str]:
        """
        Generate recommendations for improving job match
        """
        recommendations = []
        
        if match_result.overall_match_score < 0.7:
            recommendations.append("Consider developing missing technical skills")
        
        if len(match_result.missing_skills) > 0:
            top_missing = match_result.missing_skills[:3]
            recommendations.append(f"Focus on acquiring: {', '.join(top_missing)}")
        
        if not resume_exp.leadership_experience and 'leadership' in job_reqs.soft_skills_required:
            recommendations.append("Gain leadership experience through projects or team roles")
        
        return recommendations
    
    def _assess_overall_fit(self, match_result: SkillsMatchResult, 
                          resume_exp: ExperienceProfile, 
                          job_reqs: JobRequirements) -> str:
        """
        Assess overall fit for the position
        """
        score = match_result.overall_match_score
        
        if score >= 0.8:
            return "Excellent fit - Strong candidate for this role"
        elif score >= 0.6:
            return "Good fit - Candidate meets most requirements"
        elif score >= 0.4:
            return "Moderate fit - Some skill gaps to address"
        else:
            return "Limited fit - Significant skill development needed"
    
    def _save_model(self) -> None:
        """
        Save the trained model and associated components
        """
        try:
            model_data = {
                'model': self.ml_model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self) -> None:
        """
        Load pre-trained model and associated components
        """
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.ml_model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_importance = model_data.get('feature_importance', {})
                logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Flask API for web integration
app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None

@app.route('/api/predict-salary', methods=['POST'])
def api_predict_salary():
    """
    API endpoint for salary prediction
    """
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        location = data.get('location', 'bangalore')
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
        
        result = predictor.predict_salary(resume_text, location)
        return jsonify(asdict(result))
        
    except Exception as e:
        logger.error(f"Error in salary prediction API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match-job', methods=['POST'])
def api_match_job():
    """
    API endpoint for job description matching
    """
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        job_description = data.get('job_description', '')
        
        if not resume_text or not job_description:
            return jsonify({'error': 'Both resume text and job description are required'}), 400
        
        result = predictor.match_job_description(resume_text, job_description)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in job matching API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.ml_model is not None,
        'semantic_extractor_available': predictor.semantic_extractor is not None,
        'timestamp': datetime.now().isoformat()
    })

def initialize_predictor(gemini_api_key: Optional[str] = None):
    """
    Initialize the global predictor instance
    """
    global predictor
    predictor = EnhancedSalaryPredictor(gemini_api_key=gemini_api_key)
    logger.info("Enhanced Salary Predictor initialized")

if __name__ == '__main__':
    # Initialize predictor
    initialize_predictor()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)