#!/usr/bin/env python3
"""
OTS Salary Calculator Service

This module provides comprehensive salary calculation functionality for OTS Solutions.
It handles experience-based calculations, premium college weightage, role multipliers,
location adjustments, and skill-based bonuses.

Author: OTS Solutions Development Team
Version: 1.0.0
Last Updated: 2025-01-25
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from premium_college_database import PremiumCollegeDatabase
from city_multiplier_config import CityMultiplierConfig
from resume_validator import ResumeContentValidator, ValidationResult
from dynamic_salary_range import DynamicSalaryRangeCalculator


"""
OTS Salary Calculator Service

This module provides comprehensive salary calculation functionality for OTS Solutions.
It handles experience-based calculations, premium college weightage, role multipliers,
location adjustments, and skill-based bonuses.

Author: OTS Solutions Development Team
Version: 1.0.0
Last Updated: 2025-01-25
"""

# --- SalaryCalculationResult dataclass ---
@dataclass
class SalaryCalculationResult:
    """
    Data class to hold comprehensive salary calculation results.
    
    Attributes:
        base_salary: Base salary before any multipliers
        college_multiplier: Premium college multiplier applied
        role_multiplier: Role-specific multiplier applied
        location_multiplier: Location-based multiplier applied
        skill_multiplier: Skill-based multiplier applied
        final_salary: Final calculated salary after all multipliers
        experience_band: Experience band classification
        college_tier: College tier classification
        breakdown: Detailed breakdown of calculation factors
    """
    base_salary: float
    college_multiplier: float
    role_multiplier: float
    location_multiplier: float
    skill_multiplier: float
    final_salary: float
    experience_band: str
    college_tier: Optional[str]
    breakdown: Dict[str, Any]

# --- OTSSalaryCalculator class ---
class OTSSalaryCalculator:
    """
    Main salary calculator class for OTS Solutions.
    Handles all aspects of salary calculation including:
    - Experience-based base salary determination
    - Premium college weightage with time decay
    - Role-specific multipliers
    - Location-based adjustments
    - Skill-based bonuses
    - Growth projections
    """
    
    def __init__(self, config_path: str = "ots_salary_config.json", database_path: str = "ots_synthetic_database.json"):
        """
        Initialize the salary calculator with configuration.
        
        Args:
            config_path: Path to the salary configuration JSON file
            database_path: Path to the synthetic database file
        """
        self.config_path = Path(config_path)
        self.database_path = database_path
        self.config = self._load_config()
        self.college_db = PremiumCollegeDatabase(database_path)
        self.city_config = CityMultiplierConfig()
        
        # Initialize dynamic salary range calculator
        self.range_calculator = DynamicSalaryRangeCalculator(config_path)
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load salary configuration from JSON file.
        
        Returns:
            Dictionary containing salary configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Salary configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")
    
    def extract_experience_years(self, resume_text: str) -> float:
        """
        Extract years of experience from resume text.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            Number of years of experience (float for partial years)
        """
        # Pattern to match experience mentions
        experience_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*experience',
            r'total\s*experience[:\s]*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        ]
        
        experience_years = []
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text.lower())
            for match in matches:
                try:
                    years = float(match)
                    if 0 <= years <= 30:  # Reasonable range
                        experience_years.append(years)
                except ValueError:
                    continue
        
        # If no explicit experience found, try to calculate from work history
        if not experience_years:
            experience_years.append(self._calculate_experience_from_dates(resume_text))
        
        # Return the maximum reasonable experience found
        return max(experience_years) if experience_years else 0.0
    
    def _calculate_experience_from_dates(self, resume_text: str) -> float:
        """
        Calculate experience from employment dates in resume.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            Calculated years of experience
        """
        # Pattern to match date ranges (e.g., "2020-2023", "Jan 2020 - Present")
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|present|current)',
            r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|present|current)',
        ]
        
        total_months = 0
        current_year = datetime.now().year
        
        for pattern in date_patterns:
            matches = re.findall(pattern, resume_text.lower())
            for start, end in matches:
                try:
                    # Extract start year
                    start_year = int(re.search(r'\d{4}', start).group())
                    
                    # Extract end year
                    if 'present' in end or 'current' in end:
                        end_year = current_year
                    else:
                        end_year = int(re.search(r'\d{4}', end).group())
                    
                    # Calculate months (approximate)
                    if end_year >= start_year:
                        months = (end_year - start_year) * 12
                        total_months += months
                        
                except (AttributeError, ValueError):
                    continue
        
        return round(total_months / 12, 1) if total_months > 0 else 0.0
    
    def determine_experience_band(self, years_experience: float) -> str:
        """
        Determine experience band based on years of experience.
        
        Args:
            years_experience: Number of years of experience
            
        Returns:
            Experience band key (freshers, junior, mid_level, senior)
        """
        if years_experience <= 1:
            return "freshers"
        elif years_experience <= 3:
            return "junior"
        elif years_experience <= 6:
            return "mid_level"
        else:
            return "senior"
    
    def calculate_base_salary(self, years_experience: float) -> Tuple[float, str]:
        """
        Calculate base salary based on years of experience.
        
        Args:
            years_experience: Number of years of experience
            
        Returns:
            Tuple of (base_salary, experience_band)
        """
        experience_band = self.determine_experience_band(years_experience)
        band_config = self.config['salary_bands'][experience_band]
        
        base_min = band_config['base_min']
        base_max = band_config['base_max']
        
        # Linear interpolation within the band based on experience
        if experience_band == "freshers":
            # For freshers, use position within 0-1 year range
            position = min(years_experience, 1.0)
        elif experience_band == "junior":
            # For junior, use position within 1-3 year range
            position = min((years_experience - 1) / 2, 1.0)
        elif experience_band == "mid_level":
            # For mid-level, use position within 3-6 year range
            position = min((years_experience - 3) / 3, 1.0)
        else:  # senior
            # For senior, use position within 6+ year range (cap at 15 years)
            position = min((years_experience - 6) / 9, 1.0)
        
        base_salary = base_min + (base_max - base_min) * position
        return round(base_salary, 2), experience_band
    
    def extract_college_info(self, resume_text: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract college information and graduation year from resume.
        Enhanced to use premium college database for better matching.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            Tuple of (college_name, graduation_year)
        """
        college_name = None
        graduation_year = None
        
        # Enhanced college extraction using premium database fuzzy matching
        resume_lower = resume_text.lower()
        
        # First try premium college database fuzzy matching
        if self.college_db:
            # Extract potential college names using common patterns
            college_patterns = [
                r'(?:bachelor|b\.?tech|b\.?e|master|m\.?tech|m\.?e|phd|diploma).*?(?:from|at|,)\s*([^,\n()]+?)(?:,|\(|\n|$)',
                r'(?:university|institute|college|iit|nit|iiit)\s+[^,\n()]+',
                r'indian\s+institute\s+of\s+technology[^,\n()]*',
                r'national\s+institute\s+of\s+technology[^,\n()]*',
                r'iit\s+[a-z]+',
                r'nit\s+[a-z]+'
            ]
            
            potential_colleges = set()
            for pattern in college_patterns:
                matches = re.findall(pattern, resume_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, str):
                        potential_colleges.add(match.strip())
            
            # Try to find matches in premium database
            for potential_college in potential_colleges:
                if len(potential_college) > 3:  # Avoid very short matches
                    college_data = self.college_db.find_college(potential_college)
                    if college_data:
                        college_name = college_data['name']
                        break
        
        # Fallback to original method if database search fails
        if not college_name:
            for tier, tier_config in self.config['premium_colleges'].items():
                for institution in tier_config['institutions']:
                    if institution.lower() in resume_lower:
                        college_name = institution
                        break
                if college_name:
                    break
        
        # Extract graduation year
        grad_patterns = [
            r'graduated?\s*(?:in\s*)?(\d{4})',
            r'graduation\s*(?:year\s*)?[:\s]*(\d{4})',
            r'(\d{4})\s*(?:graduate|graduation)',
            r'b\.?tech\s*(?:.*?)\s*(\d{4})',
            r'bachelor\s*(?:.*?)\s*(\d{4})',
        ]
        
        for pattern in grad_patterns:
            matches = re.findall(pattern, resume_lower)
            for match in matches:
                try:
                    year = int(match)
                    if 1990 <= year <= datetime.now().year:
                        graduation_year = year
                        break
                except ValueError:
                    continue
            if graduation_year:
                break
        
        return college_name, graduation_year
    
    def calculate_college_multiplier(self, college_name: Optional[str], 
                                   graduation_year: Optional[int]) -> Tuple[float, Optional[str]]:
        """
        Calculate college premium multiplier with time decay.
        
        Args:
            college_name: Name of the college/institution
            graduation_year: Year of graduation
            
        Returns:
            Tuple of (multiplier, tier)
        """
        if not college_name:
            return 1.0, None
        
        # Find college tier
        college_tier = None
        tier_config = None
        
        for tier, config in self.config['premium_colleges'].items():
            if college_name in config['institutions']:
                college_tier = tier
                tier_config = config
                break
        
        if not tier_config:
            return 1.0, None
        
        base_multiplier = tier_config['multiplier']
        
        # Apply time decay if graduation year is available
        if graduation_year:
            years_since_graduation = datetime.now().year - graduation_year
            decay_rate = tier_config['decay_rate']
            max_decay_years = tier_config['max_decay_years']
            
            # Apply decay only up to max_decay_years
            effective_years = min(years_since_graduation, max_decay_years)
            decay_factor = 1 - (effective_years * decay_rate)
            
            # Ensure multiplier doesn't go below 1.0
            final_multiplier = max(base_multiplier * decay_factor, 1.0)
        else:
            final_multiplier = base_multiplier
        
        return round(final_multiplier, 3), college_tier
    
    def _calculate_college_premium_enhanced(self, college_name: Optional[str], experience_years: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate enhanced college premium multiplier using premium college database
        
        Args:
            college_name (Optional[str]): Name of the college
            experience_years (float): Years of experience for decay calculation
            
        Returns:
            Tuple[float, Dict[str, Any]]: College premium multiplier and college information
        """
        college_info = {
            "found_in_database": False,
            "tier": "unknown",
            "base_weightage": 1.0,
            "experience_adjusted_weightage": 1.0,
            "fallback_used": False
        }
        
        if not college_name:
            return 1.0, college_info
        
        try:
            # Try to get college information from premium database
            if self.college_db:
                college_data = self.college_db.find_college(college_name)
                if college_data:
                    college_info["found_in_database"] = True
                    college_info["tier"] = college_data.get("tier", "unknown")
                    
                    # Get experience-adjusted weightage
                    weightage, weightage_info = self.college_db.calculate_college_weightage(college_name, experience_years)
                    college_info["base_weightage"] = college_data.get("weightage", 1.0)
                    college_info["experience_adjusted_weightage"] = weightage
                    college_info["weightage_details"] = weightage_info
                    
                    return weightage, college_info
        except Exception as e:
            pass
        
        # Fallback to original calculation method
        college_info["fallback_used"] = True
        fallback_multiplier = self._calculate_college_premium_fallback(college_name, experience_years)
        college_info["experience_adjusted_weightage"] = fallback_multiplier
        
        return fallback_multiplier, college_info
    
    def _calculate_college_premium_fallback(self, college_name: Optional[str], experience_years: float) -> float:
        """
        Fallback college premium calculation method (original implementation)
        
        Args:
            college_name (Optional[str]): Name of the college
            experience_years (float): Years of experience for decay calculation
            
        Returns:
            float: College premium multiplier
        """
        if not college_name:
            return 1.0
            
        # Normalize college name for matching
        normalized_name = college_name.lower().strip()
        
        # Check premium colleges configuration
        premium_colleges = self.config.get("premium_colleges", {})
        
        # Find matching college
        college_tier = None
        for tier, config in premium_colleges.items():
            if any(normalized_name in college.lower() for college in config.get('institutions', [])):
                college_tier = tier
                break
        
        if not college_tier:
            return 1.0  # No premium for non-premium colleges
        
        # Get base multiplier for tier
        tier_multipliers = {
            "tier1": 1.3,
            "tier2": 1.2,
            "tier3": 1.1
        }
        
        base_multiplier = tier_multipliers.get(college_tier, 1.0)
        
        # Apply experience decay (college impact reduces over time)
        decay_factor = max(0.5, 1.0 - (experience_years * 0.05))  # 5% decay per year, min 50%
        
        final_multiplier = 1.0 + ((base_multiplier - 1.0) * decay_factor)
        
        return final_multiplier
    
    def extract_role_info(self, resume_text: str) -> str:
        """
        Extract role information from resume text.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            Role key for multiplier lookup
        """
        resume_lower = resume_text.lower()
        
        # Role keywords mapping
        role_keywords = {
            'technical_lead': ['technical lead', 'tech lead', 'team lead', 'lead developer'],
            'senior_developer': ['senior developer', 'senior software', 'sr developer', 'sr software'],
            'specialist': ['specialist', 'architect', 'principal', 'expert'],
            'project_manager': ['project manager', 'program manager', 'pm', 'scrum master'],
            'full_stack_developer': ['full stack', 'fullstack', 'full-stack'],
            'backend_developer': ['backend', 'back-end', 'server side'],
            'frontend_developer': ['frontend', 'front-end', 'ui developer', 'react developer'],
            'devops_engineer': ['devops', 'dev ops', 'infrastructure', 'cloud engineer'],
            'data_scientist': ['data scientist', 'data analyst', 'ml engineer', 'ai engineer'],
            'qa_engineer': ['qa engineer', 'test engineer', 'quality assurance', 'sdet']
        }
        
        # Check for role keywords in resume
        for role, keywords in role_keywords.items():
            for keyword in keywords:
                if keyword in resume_lower:
                    return role
        
        # Default to backend developer if no specific role found
        return 'backend_developer'
    
    def extract_location(self, resume_text: str) -> str:
        """
        Extract location information from resume text using the city multiplier configuration.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            Detected city name or 'other' if not found
        """
        resume_lower = resume_text.lower()
        
        # Use the city configuration to detect location
        detected_city = self.city_config.detect_city_from_text(resume_text)
        
        return detected_city if detected_city else 'other'
    
    def extract_skills(self, resume_text: str) -> List[str]:
        """
        Extract skills from resume text.
        
        Args:
            resume_text: Raw resume text content
            
        Returns:
            List of identified skills
        """
        resume_lower = resume_text.lower()
        identified_skills = []
        
        # Combine all skill categories
        all_skills = []
        for category in self.config['skill_multipliers'].values():
            all_skills.extend([skill.lower() for skill in category['skills']])
        
        # Check for skills in resume
        for skill in all_skills:
            if skill.lower() in resume_lower:
                identified_skills.append(skill)
        
        return identified_skills
    
    def calculate_skill_multiplier(self, skills: List[str]) -> float:
        """
        Calculate skill-based multiplier.
        
        Args:
            skills: List of identified skills
            
        Returns:
            Skill multiplier value
        """
        total_multiplier = 1.0
        skills_lower = [skill.lower() for skill in skills]
        
        for category, config in self.config['skill_multipliers'].items():
            category_skills = [skill.lower() for skill in config['skills']]
            matching_skills = set(skills_lower) & set(category_skills)
            
            if matching_skills:
                # Apply multiplier based on number of matching skills in category
                skill_bonus = (config['multiplier'] - 1.0) * min(len(matching_skills) / len(category_skills), 1.0)
                total_multiplier += skill_bonus
        
        return round(total_multiplier, 3)
    
    def calculate_salary(self, resume_text: str) -> SalaryCalculationResult:
        """
        Calculate comprehensive salary based on resume analysis with enhanced premium college weightage.
        This method validates the resume, extracts all relevant information, applies all multipliers, and returns a detailed result.
        Args:
            resume_text: Raw resume text content
        Returns:
            SalaryCalculationResult object with detailed breakdown
        """
        # Initialize resume validator if not already done
        if not hasattr(self, 'resume_validator'):
            self.resume_validator = ResumeContentValidator()
        
        # Validate resume content before processing
        validation_result = self.resume_validator.validate_resume(resume_text)
        # Check if document is a valid resume
        if not validation_result.is_valid_resume:
            # Return error result for non-resume documents
            return SalaryCalculationResult(
                base_salary=0.0,
                college_multiplier=0.0,
                role_multiplier=0.0,
                location_multiplier=0.0,
                skill_multiplier=0.0,
                final_salary=0.0,
                experience_band="invalid",
                college_tier=None,
                breakdown={
                    'validation_error': True,
                    'validation_result': validation_result.validation_result.value,
                    'document_type': validation_result.document_type.value,
                    'confidence_score': validation_result.confidence_score,
                    'quality_score': validation_result.quality_score,
                    'recommendations': validation_result.recommendations,
                    'error_message': f"Document is not a valid resume. Detected as: {validation_result.document_type.value}"
                }
            )
        
        # Log validation success
        print(f"Resume validation passed - Confidence: {validation_result.confidence_score:.2f}, Quality: {validation_result.quality_score:.2f}")
        
        # Extract information from resume
        years_experience = self.extract_experience_years(resume_text)
        college_name, graduation_year = self.extract_college_info(resume_text)
        role = self.extract_role_info(resume_text)
        location = self.extract_location(resume_text)
        skills = self.extract_skills(resume_text)
        # Calculate base salary
        base_salary, experience_band = self.calculate_base_salary(years_experience)
        # Calculate multipliers with enhanced premium college weightage
        college_multiplier, college_tier = self.calculate_college_multiplier(college_name, graduation_year)
        # Apply enhanced premium college weightage from database
        enhanced_info = None
        if college_name:
            enhanced_multiplier, enhanced_info = self._calculate_college_premium_enhanced(college_name, years_experience)
            if enhanced_multiplier > college_multiplier:
                college_multiplier = enhanced_multiplier
            # Update college_tier from enhanced database if available
            if enhanced_info and enhanced_info.get('found_in_database'):
                college_tier = enhanced_info.get('tier', college_tier)
        role_multiplier = self.config['role_multipliers'].get(role, {}).get('multiplier', 1.0)
        # Calculate location multiplier using city configuration
        location_multiplier, city_info = self.city_config.get_city_multiplier(location)
        # Add tech hub bonus if applicable
        tech_hub_bonus = self.city_config.get_tech_hub_bonus(location)
        if tech_hub_bonus > 0:
            location_multiplier += tech_hub_bonus
        skill_multiplier = self.calculate_skill_multiplier(skills)
        # Calculate final salary
        final_salary = (
            base_salary * 
            college_multiplier * 
            role_multiplier * 
            location_multiplier * 
            skill_multiplier
        )
        # Apply validation rules
        min_cap = self.config['validation_rules']['min_salary_cap']
        max_cap = self.config['validation_rules']['max_salary_cap']
        final_salary = max(min_cap, min(final_salary, max_cap))
        # Create detailed breakdown with premium college and city info
        breakdown = {
            'years_experience': years_experience,
            'college_name': college_name,
            'graduation_year': graduation_year,
            'role': role,
            'location': location,
            'skills': skills,
            'premium_college_info': {
                 'tier': college_tier,
                 'database_weightage': self.college_db.calculate_college_weightage(college_name, years_experience)[0] if college_name else 1.0,
                 'final_multiplier': college_multiplier
             },
            'city_info': {
                'detected_city': location,
                'city_tier': city_info.tier.value if city_info else 'Unknown',
                'state': city_info.state if city_info else 'Unknown',
                'base_multiplier': location_multiplier - tech_hub_bonus if tech_hub_bonus > 0 else location_multiplier,
                'tech_hub_bonus': tech_hub_bonus,
                'final_location_multiplier': location_multiplier,
                'is_tech_hub': city_info.tech_hub if city_info else False,
                'population': city_info.population if city_info else 'Unknown'
            },
            'calculation_steps': {
                'base_salary': base_salary,
                'college_multiplier': college_multiplier,
                'role_multiplier': role_multiplier,
                'location_multiplier': location_multiplier,
                'skill_multiplier': skill_multiplier,
                'before_caps': base_salary * college_multiplier * role_multiplier * location_multiplier * skill_multiplier,
                'final_salary': final_salary
            }
        }
        return SalaryCalculationResult(
            base_salary=round(base_salary, 2),
            college_multiplier=college_multiplier,
            role_multiplier=role_multiplier,
            location_multiplier=location_multiplier,
            skill_multiplier=skill_multiplier,
            final_salary=round(final_salary, 2),
            experience_band=experience_band,
            college_tier=college_tier,
            breakdown=breakdown
        )
    
    def calculate_dynamic_salary_range(self, 
                                      base_salary: float,
                                      experience_band: str,
                                      skills: List[str],
                                      location: str,
                                      college_tier: Optional[str] = None,
                                      college_multiplier: float = 1.0) -> Dict[str, Any]:
        """
        Calculate dynamic salary range using the advanced range calculator.
        
        Args:
            base_salary: Base calculated salary
            experience_band: Experience level (freshers, junior, mid_level, senior)
            skills: List of identified skills
            location: Location/city
            college_tier: College tier (if applicable)
            college_multiplier: College premium multiplier
            
        Returns:
            Dictionary containing dynamic range calculation results
        """
        try:
            # Use the dynamic range calculator
            range_result = self.range_calculator.calculate_dynamic_range(
                base_salary=base_salary,
                experience_band=experience_band,
                skills=skills,
                location=location,
                college_tier=college_tier,
                college_multiplier=college_multiplier
            )
            
            # Convert to dictionary format for API response
            return {
                "min": range_result.min_salary,
                "max": range_result.max_salary,
                "median": range_result.median_salary,
                "range_width": range_result.range_width,
                "confidence_level": range_result.confidence_level,
                "market_factors": range_result.market_factors,
                "calculation_breakdown": range_result.calculation_breakdown,
                "recommendations": range_result.recommendations
            }
            
        except Exception as e:
            # Fallback to simple range calculation
            return self._get_simple_fallback_range(base_salary, experience_band)
    
    def _get_simple_fallback_range(self, base_salary: float, experience_band: str) -> Dict[str, Any]:
        """
        Get simple fallback salary range if dynamic calculation fails.
        
        Args:
            base_salary: Base calculated salary
            experience_band: Experience level
            
        Returns:
            Simple range dictionary
        """
        # Use the simple range method from dynamic calculator
        simple_range = self.range_calculator.get_simple_range(base_salary, experience_band)
        
        return {
            "min": simple_range["min"],
            "max": simple_range["max"],
            "median": base_salary * 1.1,
            "range_width": simple_range["max"] - simple_range["min"],
            "confidence_level": 0.6,
            "market_factors": {"fallback": True},
            "calculation_breakdown": {"method": "simple_fallback"},
            "recommendations": ["Using simplified range calculation due to processing error"]
        }
    
    def project_salary_growth(self, current_salary: float, years: int = 5, 
                             current_experience: float = 0, current_role: str = "software_engineer") -> Dict[str, Any]:
        """
        Enhanced salary growth projection with role transition factors and experience-based growth.
        
        Args:
            current_salary: Current calculated salary
            years: Number of years to project (default: 5)
            current_experience: Current years of experience
            current_role: Current role for transition planning
            
        Returns:
            Dictionary with detailed year-wise salary projections and role transitions
        """
        growth_config = self.config['calculation_rules']['growth_projection']
        yearly_growth = growth_config['yearly_growth_rate']
        promotion_boost = growth_config['promotion_boost']
        max_years = min(years, growth_config['max_projection_years'])
        
        # Role transition mapping based on experience and career progression
        role_transitions = {
            "software_engineer": {
                3: {"role": "senior_software_engineer", "boost": 0.25},
                5: {"role": "tech_lead", "boost": 0.40},
                7: {"role": "engineering_manager", "boost": 0.60}
            },
            "senior_software_engineer": {
                2: {"role": "tech_lead", "boost": 0.20},
                4: {"role": "engineering_manager", "boost": 0.35},
                6: {"role": "senior_manager", "boost": 0.50}
            },
            "tech_lead": {
                2: {"role": "engineering_manager", "boost": 0.25},
                4: {"role": "senior_manager", "boost": 0.40},
                6: {"role": "director", "boost": 0.60}
            }
        }
        
        # Experience-based growth rate adjustments
        def get_experience_growth_factor(exp_years: float) -> float:
            if exp_years < 2:
                return 1.2  # High growth for junior developers
            elif exp_years < 5:
                return 1.0  # Standard growth
            elif exp_years < 10:
                return 0.8  # Slower growth as experience increases
            else:
                return 0.6  # Minimal growth for very senior roles
        
        projections = {
            'current': {
                'salary': current_salary,
                'role': current_role,
                'experience': current_experience
            },
            'yearly_breakdown': {},
            'role_transitions': [],
            'growth_factors': {
                'base_yearly_growth': yearly_growth,
                'promotion_boost': promotion_boost
            }
        }
        
        current_role_working = current_role
        
        for year in range(1, max_years + 1):
            projected_experience = current_experience + year
            
            # Calculate base growth with experience factor
            experience_factor = get_experience_growth_factor(projected_experience)
            adjusted_growth_rate = yearly_growth * experience_factor
            
            # Apply yearly growth
            projected_salary = current_salary * ((1 + adjusted_growth_rate) ** year)
            
            # Check for role transitions
            role_boost = 0
            new_role = current_role_working
            
            if current_role_working in role_transitions:
                for transition_year, transition_info in role_transitions[current_role_working].items():
                    if year == transition_year:
                        role_boost = transition_info['boost']
                        new_role = transition_info['role']
                        current_role_working = new_role
                        
                        projections['role_transitions'].append({
                            'year': year,
                            'from_role': current_role_working if year == transition_year else current_role,
                            'to_role': new_role,
                            'salary_boost': role_boost,
                            'experience_years': projected_experience
                        })
                        break
            
            # Apply role transition boost
            if role_boost > 0:
                projected_salary *= (1 + role_boost)
            
            # Apply standard promotion boost every 2-3 years (if no role transition)
            elif year % 3 == 0:
                projected_salary *= (1 + promotion_boost)
            
            projections['yearly_breakdown'][f'year_{year}'] = {
                'salary': round(projected_salary, 2),
                'role': new_role,
                'experience': projected_experience,
                'growth_rate_applied': adjusted_growth_rate,
                'role_boost_applied': role_boost,
                'total_growth_from_current': round(((projected_salary - current_salary) / current_salary) * 100, 1)
            }
        
        return projections


# Example usage and testing
if __name__ == "__main__":
    # Initialize calculator
    calculator = OTSSalaryCalculator()
    
    # Sample resume text for testing
    sample_resume = """
    John Doe
    Senior Software Developer
    
    Education:
    B.Tech Computer Science, IIT Delhi (2018)
    
    Experience:
    5 years of experience in full-stack development
    
    Skills:
    Python, React, Node.js, AWS, Docker
    
    Location: Bangalore
    """
    
    # Calculate salary
    result = calculator.calculate_salary(sample_resume)
    
    print(f"Calculated Salary: ₹{result.final_salary} LPA")
    print(f"Experience Band: {result.experience_band}")
    print(f"College Tier: {result.college_tier}")
    print(f"Base Salary: ₹{result.base_salary} LPA")
    print(f"Multipliers - College: {result.college_multiplier}, Role: {result.role_multiplier}, Location: {result.location_multiplier}, Skills: {result.skill_multiplier}")
    
    # Project growth
    growth = calculator.project_salary_growth(result.final_salary)
    print(f"\nSalary Growth Projection:")
    for year, salary in growth.items():
        print(f"{year}: ₹{salary} LPA")