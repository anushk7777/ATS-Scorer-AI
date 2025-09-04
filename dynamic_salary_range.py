#!/usr/bin/env python3
"""
Dynamic Salary Range Calculator Module

This module provides intelligent salary range calculation based on:
1. Experience level and market variance
2. Skill demand and rarity factors
3. Location market conditions
4. Industry standards and trends
5. College tier and premium factors

Author: OTS Solutions
Version: 1.0.0
Date: 2025-01-25
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Market condition classifications"""
    HOT = "hot"           # High demand, low supply
    STABLE = "stable"     # Balanced market
    COMPETITIVE = "competitive"  # High supply, moderate demand
    SATURATED = "saturated"      # High supply, low demand

class SkillDemand(Enum):
    """Skill demand levels in the market"""
    CRITICAL = "critical"     # Very high demand, rare skills
    HIGH = "high"            # High demand, good supply
    MODERATE = "moderate"     # Moderate demand
    LOW = "low"              # Low demand, common skills

@dataclass
class SalaryRangeResult:
    """
    Data class to store dynamic salary range calculation results.
    """
    min_salary: float
    max_salary: float
    median_salary: float
    range_width: float
    confidence_level: float
    market_factors: Dict[str, Any]
    calculation_breakdown: Dict[str, Any]
    recommendations: List[str]

class DynamicSalaryRangeCalculator:
    """
    Advanced salary range calculator that provides dynamic ranges based on
    multiple market factors, experience levels, skills, and location conditions.
    """
    
    def __init__(self, config_path: str = "ots_salary_config.json"):
        """
        Initialize the dynamic salary range calculator.
        
        Args:
            config_path: Path to the salary configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Market variance factors by experience level
        self.experience_variance = {
            "freshers": {
                "base_variance": 0.25,  # ±25% base variance
                "skill_impact": 0.15,   # Skills can add ±15%
                "location_impact": 0.20, # Location can add ±20%
                "college_impact": 0.30   # College can add ±30%
            },
            "junior": {
                "base_variance": 0.30,
                "skill_impact": 0.25,
                "location_impact": 0.25,
                "college_impact": 0.20
            },
            "mid_level": {
                "base_variance": 0.35,
                "skill_impact": 0.35,
                "location_impact": 0.30,
                "college_impact": 0.15
            },
            "senior": {
                "base_variance": 0.45,
                "skill_impact": 0.50,
                "location_impact": 0.35,
                "college_impact": 0.10
            }
        }
        
        # Skill demand mapping for market analysis
        self.skill_demand_mapping = {
            # Critical demand skills (AI/ML, Cloud, etc.)
            "critical": {
                "skills": ["machine learning", "artificial intelligence", "deep learning", 
                          "kubernetes", "aws architect", "devops", "blockchain", "golang"],
                "demand_multiplier": 1.4,
                "variance_increase": 0.20
            },
            # High demand skills
            "high": {
                "skills": ["react", "node.js", "python", "aws", "docker", "microservices", 
                          "system design", "data science", "cybersecurity"],
                "demand_multiplier": 1.2,
                "variance_increase": 0.15
            },
            # Moderate demand skills
            "moderate": {
                "skills": ["java", "javascript", "sql", "angular", "spring boot", 
                          "rest api", "git", "agile"],
                "demand_multiplier": 1.0,
                "variance_increase": 0.10
            },
            # Low demand skills
            "low": {
                "skills": ["php", "jquery", "xml", "soap", "legacy systems"],
                "demand_multiplier": 0.9,
                "variance_increase": 0.05
            }
        }
        
        # Location market conditions
        self.location_market_conditions = {
            # Tier 1 cities - Hot markets
            "bangalore": {"condition": MarketCondition.HOT, "variance_multiplier": 1.3},
            "mumbai": {"condition": MarketCondition.HOT, "variance_multiplier": 1.25},
            "delhi": {"condition": MarketCondition.HOT, "variance_multiplier": 1.25},
            "gurgaon": {"condition": MarketCondition.HOT, "variance_multiplier": 1.25},
            "hyderabad": {"condition": MarketCondition.STABLE, "variance_multiplier": 1.15},
            "pune": {"condition": MarketCondition.STABLE, "variance_multiplier": 1.15},
            
            # Tier 2 cities - Stable to competitive
            "chennai": {"condition": MarketCondition.STABLE, "variance_multiplier": 1.10},
            "kolkata": {"condition": MarketCondition.COMPETITIVE, "variance_multiplier": 1.05},
            "ahmedabad": {"condition": MarketCondition.COMPETITIVE, "variance_multiplier": 1.05},
            "kochi": {"condition": MarketCondition.STABLE, "variance_multiplier": 1.08},
            
            # Default for other locations
            "default": {"condition": MarketCondition.COMPETITIVE, "variance_multiplier": 1.0}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load salary configuration from JSON file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
                self.logger.info(f"Loaded salary configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if file loading fails.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "salary_bands": {
                "freshers": {"base_min": 300000, "base_max": 800000},
                "junior": {"base_min": 600000, "base_max": 1200000},
                "mid_level": {"base_min": 1000000, "base_max": 2000000},
                "senior": {"base_min": 1800000, "base_max": 3500000}
            }
        }
    
    def calculate_dynamic_range(self, 
                              base_salary: float,
                              experience_band: str,
                              skills: List[str],
                              location: str,
                              college_tier: Optional[str] = None,
                              college_multiplier: float = 1.0,
                              market_conditions: Optional[Dict[str, Any]] = None) -> SalaryRangeResult:
        """
        Calculate dynamic salary range based on multiple factors.
        
        Args:
            base_salary: Base calculated salary
            experience_band: Experience level (freshers, junior, mid_level, senior)
            skills: List of identified skills
            location: Location/city
            college_tier: College tier (if applicable)
            college_multiplier: College premium multiplier
            market_conditions: Additional market condition data
            
        Returns:
            SalaryRangeResult with calculated range and analysis
        """
        try:
            self.logger.info(f"Calculating dynamic salary range for {experience_band} level in {location}")
            
            # Get base variance factors for experience level
            variance_config = self.experience_variance.get(experience_band, self.experience_variance["junior"])
            
            # Calculate skill-based variance
            skill_variance, skill_demand_level = self._calculate_skill_variance(skills, variance_config)
            
            # Calculate location-based variance
            location_variance, market_condition = self._calculate_location_variance(location, variance_config)
            
            # Calculate college-based variance
            college_variance = self._calculate_college_variance(college_tier, college_multiplier, variance_config)
            
            # Calculate total variance
            total_variance = self._calculate_total_variance(
                variance_config["base_variance"],
                skill_variance,
                location_variance,
                college_variance
            )
            
            # Calculate range bounds
            min_salary, max_salary = self._calculate_range_bounds(base_salary, total_variance)
            
            # Calculate median (weighted towards base salary)
            median_salary = self._calculate_median_salary(base_salary, min_salary, max_salary, skill_demand_level)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                experience_band, skills, location, college_tier
            )
            
            # Compile market factors
            market_factors = {
                "experience_level": experience_band,
                "skill_demand": skill_demand_level.value if skill_demand_level else "moderate",
                "market_condition": market_condition.value,
                "location_tier": self._get_location_tier(location),
                "college_impact": college_tier is not None
            }
            
            # Create calculation breakdown
            calculation_breakdown = {
                "base_salary": base_salary,
                "total_variance_percentage": round(total_variance * 100, 1),
                "variance_components": {
                    "base_variance": round(variance_config["base_variance"] * 100, 1),
                    "skill_variance": round(skill_variance * 100, 1),
                    "location_variance": round(location_variance * 100, 1),
                    "college_variance": round(college_variance * 100, 1)
                },
                "range_calculation": {
                    "min_multiplier": round((min_salary / base_salary), 3),
                    "max_multiplier": round((max_salary / base_salary), 3),
                    "median_multiplier": round((median_salary / base_salary), 3)
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_range_recommendations(
                total_variance, skill_demand_level, market_condition, experience_band
            )
            
            range_width = max_salary - min_salary
            
            self.logger.info(f"Dynamic range calculated: ₹{min_salary:,.0f} - ₹{max_salary:,.0f} (±{total_variance*100:.1f}%)")
            
            return SalaryRangeResult(
                min_salary=round(min_salary, 2),
                max_salary=round(max_salary, 2),
                median_salary=round(median_salary, 2),
                range_width=round(range_width, 2),
                confidence_level=confidence_level,
                market_factors=market_factors,
                calculation_breakdown=calculation_breakdown,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic salary range: {e}")
            # Return fallback range
            return self._get_fallback_range(base_salary)
    
    def _calculate_skill_variance(self, skills: List[str], variance_config: Dict[str, float]) -> Tuple[float, Optional[SkillDemand]]:
        """
        Calculate variance based on skill demand in the market.
        
        Args:
            skills: List of identified skills
            variance_config: Variance configuration for experience level
            
        Returns:
            Tuple of (skill_variance, skill_demand_level)
        """
        if not skills:
            return 0.0, SkillDemand.LOW
        
        skills_lower = [skill.lower() for skill in skills]
        max_demand_level = SkillDemand.LOW
        max_variance_increase = 0.0
        
        # Check each skill demand category
        for demand_level, demand_config in self.skill_demand_mapping.items():
            demand_skills = [skill.lower() for skill in demand_config["skills"]]
            matching_skills = set(skills_lower) & set(demand_skills)
            
            if matching_skills:
                # Update max demand level found
                current_demand = SkillDemand(demand_level)
                if self._get_demand_priority(current_demand) > self._get_demand_priority(max_demand_level):
                    max_demand_level = current_demand
                    max_variance_increase = demand_config["variance_increase"]
        
        # Calculate skill variance
        base_skill_impact = variance_config["skill_impact"]
        skill_variance = base_skill_impact + max_variance_increase
        
        return min(skill_variance, 0.6), max_demand_level  # Cap at 60%
    
    def _get_demand_priority(self, demand: SkillDemand) -> int:
        """
        Get priority value for skill demand level.
        
        Args:
            demand: SkillDemand enum value
            
        Returns:
            Priority integer (higher = more priority)
        """
        priority_map = {
            SkillDemand.LOW: 1,
            SkillDemand.MODERATE: 2,
            SkillDemand.HIGH: 3,
            SkillDemand.CRITICAL: 4
        }
        return priority_map.get(demand, 1)
    
    def _calculate_location_variance(self, location: str, variance_config: Dict[str, float]) -> Tuple[float, MarketCondition]:
        """
        Calculate variance based on location market conditions.
        
        Args:
            location: Location/city name
            variance_config: Variance configuration for experience level
            
        Returns:
            Tuple of (location_variance, market_condition)
        """
        location_lower = location.lower() if location else "default"
        
        # Get location market data
        location_data = self.location_market_conditions.get(
            location_lower, 
            self.location_market_conditions["default"]
        )
        
        market_condition = location_data["condition"]
        variance_multiplier = location_data["variance_multiplier"]
        
        # Calculate location variance
        base_location_impact = variance_config["location_impact"]
        location_variance = base_location_impact * variance_multiplier
        
        return min(location_variance, 0.5), market_condition  # Cap at 50%
    
    def _calculate_college_variance(self, college_tier: Optional[str], college_multiplier: float, variance_config: Dict[str, float]) -> float:
        """
        Calculate variance based on college tier and premium.
        
        Args:
            college_tier: College tier classification
            college_multiplier: College premium multiplier
            variance_config: Variance configuration for experience level
            
        Returns:
            College-based variance
        """
        if not college_tier or college_multiplier <= 1.0:
            return 0.0
        
        # Higher college multiplier = higher variance (more negotiation power)
        college_premium_factor = min((college_multiplier - 1.0) * 2, 1.0)  # Normalize to 0-1
        base_college_impact = variance_config["college_impact"]
        
        college_variance = base_college_impact * college_premium_factor
        
        return min(college_variance, 0.4)  # Cap at 40%
    
    def _calculate_total_variance(self, base_variance: float, skill_variance: float, 
                                location_variance: float, college_variance: float) -> float:
        """
        Calculate total variance by combining all factors.
        
        Args:
            base_variance: Base variance for experience level
            skill_variance: Skill-based variance
            location_variance: Location-based variance
            college_variance: College-based variance
            
        Returns:
            Total variance percentage
        """
        # Use weighted combination rather than simple addition
        total_variance = (
            base_variance * 0.4 +      # 40% weight to base
            skill_variance * 0.3 +     # 30% weight to skills
            location_variance * 0.2 +  # 20% weight to location
            college_variance * 0.1     # 10% weight to college
        )
        
        # Cap total variance at 70%
        return min(total_variance, 0.7)
    
    def _calculate_range_bounds(self, base_salary: float, total_variance: float) -> Tuple[float, float]:
        """
        Calculate salary range bounds based on total variance.
        
        Args:
            base_salary: Base calculated salary
            total_variance: Total variance percentage
            
        Returns:
            Tuple of (min_salary, max_salary)
        """
        # Asymmetric range - typically higher upside than downside
        downside_factor = total_variance * 0.7  # 70% of variance on downside
        upside_factor = total_variance * 1.3    # 130% of variance on upside
        
        min_salary = base_salary * (1 - downside_factor)
        max_salary = base_salary * (1 + upside_factor)
        
        # Ensure minimum bounds
        min_salary = max(min_salary, base_salary * 0.6)  # Never below 60% of base
        max_salary = max(max_salary, base_salary * 1.2)  # At least 20% above base
        
        return min_salary, max_salary
    
    def _calculate_median_salary(self, base_salary: float, min_salary: float, 
                               max_salary: float, skill_demand: SkillDemand) -> float:
        """
        Calculate median salary with bias based on skill demand.
        
        Args:
            base_salary: Base calculated salary
            min_salary: Minimum salary in range
            max_salary: Maximum salary in range
            skill_demand: Skill demand level
            
        Returns:
            Median salary
        """
        # Bias median based on skill demand
        demand_bias = {
            SkillDemand.CRITICAL: 0.7,   # Bias towards upper range
            SkillDemand.HIGH: 0.6,
            SkillDemand.MODERATE: 0.5,   # True median
            SkillDemand.LOW: 0.4         # Bias towards lower range
        }
        
        bias_factor = demand_bias.get(skill_demand, 0.5)
        median_salary = min_salary + (max_salary - min_salary) * bias_factor
        
        return median_salary
    
    def _calculate_confidence_level(self, experience_band: str, skills: List[str], 
                                  location: str, college_tier: Optional[str]) -> float:
        """
        Calculate confidence level for the salary range.
        
        Args:
            experience_band: Experience level
            skills: List of skills
            location: Location
            college_tier: College tier
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        confidence_factors = {
            'experience': 0.3 if experience_band in ["mid_level", "senior"] else 0.2,
            'skills': min(len(skills) * 0.05, 0.25),  # Up to 25% for skills
            'location': 0.2 if location.lower() in self.location_market_conditions else 0.1,
            'college': 0.15 if college_tier else 0.05
        }
        
        total_confidence = sum(confidence_factors.values())
        return min(total_confidence, 0.95)  # Cap at 95%
    
    def _get_location_tier(self, location: str) -> str:
        """
        Get location tier classification.
        
        Args:
            location: Location name
            
        Returns:
            Location tier string
        """
        tier_1_cities = ["bangalore", "mumbai", "delhi", "gurgaon"]
        tier_2_cities = ["hyderabad", "pune", "chennai", "kolkata", "ahmedabad", "kochi"]
        
        location_lower = location.lower() if location else ""
        
        if location_lower in tier_1_cities:
            return "Tier 1"
        elif location_lower in tier_2_cities:
            return "Tier 2"
        else:
            return "Tier 3+"
    
    def _generate_range_recommendations(self, total_variance: float, skill_demand: Optional[SkillDemand], 
                                      market_condition: MarketCondition, experience_band: str) -> List[str]:
        """
        Generate recommendations based on range analysis.
        
        Args:
            total_variance: Total calculated variance
            skill_demand: Skill demand level
            market_condition: Market condition
            experience_band: Experience level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Variance-based recommendations
        if total_variance > 0.5:
            recommendations.append("High salary variance indicates strong negotiation potential")
        elif total_variance < 0.2:
            recommendations.append("Limited salary variance suggests standardized compensation")
        
        # Skill-based recommendations
        if skill_demand == SkillDemand.CRITICAL:
            recommendations.append("Critical skills detected - aim for upper range of salary band")
        elif skill_demand == SkillDemand.HIGH:
            recommendations.append("High-demand skills provide good negotiation leverage")
        elif skill_demand == SkillDemand.LOW:
            recommendations.append("Consider developing in-demand skills to increase salary potential")
        
        # Market-based recommendations
        if market_condition == MarketCondition.HOT:
            recommendations.append("Hot job market - consider targeting above-median offers")
        elif market_condition == MarketCondition.SATURATED:
            recommendations.append("Competitive market - focus on unique value proposition")
        
        # Experience-based recommendations
        if experience_band == "freshers":
            recommendations.append("Focus on learning opportunities and skill development")
        elif experience_band == "senior":
            recommendations.append("Senior level - emphasize leadership and strategic impact")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def _get_fallback_range(self, base_salary: float) -> SalaryRangeResult:
        """
        Get fallback salary range if calculation fails.
        
        Args:
            base_salary: Base salary
            
        Returns:
            Fallback SalaryRangeResult
        """
        min_salary = base_salary * 0.8
        max_salary = base_salary * 1.4
        median_salary = base_salary * 1.1
        
        return SalaryRangeResult(
            min_salary=min_salary,
            max_salary=max_salary,
            median_salary=median_salary,
            range_width=max_salary - min_salary,
            confidence_level=0.5,
            market_factors={"fallback": True},
            calculation_breakdown={"error": "Fallback calculation used"},
            recommendations=["Unable to calculate detailed range - using conservative estimates"]
        )
    
    def get_simple_range(self, base_salary: float, experience_band: str) -> Dict[str, float]:
        """
        Get a simple salary range for quick calculations.
        
        Args:
            base_salary: Base calculated salary
            experience_band: Experience level
            
        Returns:
            Dictionary with min and max salary
        """
        variance_map = {
            "freshers": 0.25,
            "junior": 0.30,
            "mid_level": 0.35,
            "senior": 0.45
        }
        
        variance = variance_map.get(experience_band, 0.30)
        
        return {
            "min": base_salary * (1 - variance * 0.7),
            "max": base_salary * (1 + variance * 1.3)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize calculator
    range_calculator = DynamicSalaryRangeCalculator()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Senior Developer with Critical Skills in Bangalore",
            "base_salary": 2500000,
            "experience_band": "senior",
            "skills": ["machine learning", "kubernetes", "aws", "python", "system design"],
            "location": "bangalore",
            "college_tier": "Tier 1",
            "college_multiplier": 1.4
        },
        {
            "name": "Fresher with Moderate Skills in Pune",
            "base_salary": 500000,
            "experience_band": "freshers",
            "skills": ["java", "spring boot", "sql"],
            "location": "pune",
            "college_tier": None,
            "college_multiplier": 1.0
        },
        {
            "name": "Mid-level Developer with High-demand Skills in Delhi",
            "base_salary": 1500000,
            "experience_band": "mid_level",
            "skills": ["react", "node.js", "aws", "docker", "microservices"],
            "location": "delhi",
            "college_tier": "Tier 2",
            "college_multiplier": 1.2
        }
    ]
    
    print("=== Dynamic Salary Range Calculator Test ===")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        
        result = range_calculator.calculate_dynamic_range(
            base_salary=scenario['base_salary'],
            experience_band=scenario['experience_band'],
            skills=scenario['skills'],
            location=scenario['location'],
            college_tier=scenario['college_tier'],
            college_multiplier=scenario['college_multiplier']
        )
        
        print(f"   Base Salary: ₹{scenario['base_salary']:,}")
        print(f"   Salary Range: ₹{result.min_salary:,.0f} - ₹{result.max_salary:,.0f}")
        print(f"   Median: ₹{result.median_salary:,.0f}")
        print(f"   Variance: ±{result.calculation_breakdown['total_variance_percentage']}%")
        print(f"   Confidence: {result.confidence_level:.1%}")
        print(f"   Market Factors: {result.market_factors}")
        print(f"   Top Recommendation: {result.recommendations[0] if result.recommendations else 'None'}")
    
    print("\n=== Test Completed Successfully ===")