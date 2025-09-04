#!/usr/bin/env python3
"""
OTS Synthetic Database Generator
Creates realistic synthetic data for testing the OTS Salary Prediction System

This module generates:
- Premium college catalog with tier classifications
- Salary structures based on experience levels
- Role-based compensation data
- Location-specific salary adjustments
- Skill-based premium calculations

Author: OTS Development Team
Version: 1.0.0
Date: 2025-01-15
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import os

class OTSSyntheticDatabase:
    """
    Generates comprehensive synthetic data for OTS salary prediction system
    
    This class creates realistic datasets that mirror real-world salary structures,
    premium college classifications, and compensation patterns in the Indian IT industry.
    """
    
    def __init__(self, config_path: str = "ots_salary_config.json"):
        """
        Initialize the synthetic database generator
        
        Args:
            config_path (str): Path to the OTS salary configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.synthetic_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_records": 0,
                "data_sources": ["synthetic_generation", "industry_benchmarks"]
            },
            "premium_colleges": {},
            "salary_records": [],
            "role_compensation": {},
            "location_data": {},
            "skill_premiums": {},
            "growth_patterns": {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the OTS salary configuration
        
        Returns:
            Dict[str, Any]: Configuration data
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using default values.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Provide default configuration if config file is not found
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "salary_bands": {
                "freshers": {"base": 400000, "premium_multiplier": 1.3},
                "junior": {"base": 800000, "premium_multiplier": 1.25},
                "mid_level": {"base": 1500000, "premium_multiplier": 1.2},
                "senior": {"base": 2500000, "premium_multiplier": 1.15}
            },
            "premium_colleges": {
                "tier_1": {"multiplier": 1.3, "institutions": []},
                "tier_2": {"multiplier": 1.15, "institutions": []},
                "tier_3": {"multiplier": 1.05, "institutions": []}
            }
        }
    
    def generate_premium_college_catalog(self) -> Dict[str, Any]:
        """
        Generate comprehensive premium college catalog with realistic data
        
        Returns:
            Dict[str, Any]: Premium college catalog
        """
        # Tier 1 Institutions (Top-tier engineering colleges)
        tier_1_colleges = [
            "Indian Institute of Technology Delhi",
            "Indian Institute of Technology Bombay",
            "Indian Institute of Technology Madras",
            "Indian Institute of Technology Kanpur",
            "Indian Institute of Technology Kharagpur",
            "Indian Institute of Technology Roorkee",
            "Indian Institute of Technology Guwahati",
            "Indian Institute of Science Bangalore",
            "Delhi Technological University",
            "Netaji Subhas University of Technology",
            "Birla Institute of Technology and Science Pilani",
            "Vellore Institute of Technology",
            "Manipal Institute of Technology",
            "National Institute of Technology Trichy",
            "National Institute of Technology Warangal"
        ]
        
        # Tier 2 Institutions (Good engineering colleges)
        tier_2_colleges = [
            "Jadavpur University",
            "Anna University",
            "Pune Institute of Computer Technology",
            "College of Engineering Pune",
            "PSG College of Technology",
            "Thiagarajar College of Engineering",
            "National Institute of Technology Calicut",
            "National Institute of Technology Surathkal",
            "Motilal Nehru National Institute of Technology",
            "Malaviya National Institute of Technology",
            "Sardar Vallabhbhai National Institute of Technology",
            "Visvesvaraya National Institute of Technology",
            "Indian Institute of Information Technology Allahabad",
            "Indian Institute of Information Technology Hyderabad",
            "Dhirubhai Ambani Institute of Information and Communication Technology"
        ]
        
        # Tier 3 Institutions (Decent engineering colleges)
        tier_3_colleges = [
            "SRM Institute of Science and Technology",
            "Amity University",
            "Lovely Professional University",
            "Kalinga Institute of Industrial Technology",
            "Chandigarh University",
            "Thapar Institute of Engineering and Technology",
            "Jaypee Institute of Information Technology",
            "Bharati Vidyapeeth College of Engineering",
            "Ramaiah Institute of Technology",
            "BMS College of Engineering",
            "RV College of Engineering",
            "PES University",
            "Dayananda Sagar College of Engineering",
            "Christ University",
            "Symbiosis Institute of Technology"
        ]
        
        # Generate detailed college data
        premium_colleges = {
            "tier_1": {
                "multiplier": 1.30,
                "decay_rate": 0.02,
                "max_years": 10,
                "institutions": {}
            },
            "tier_2": {
                "multiplier": 1.15,
                "decay_rate": 0.03,
                "max_years": 8,
                "institutions": {}
            },
            "tier_3": {
                "multiplier": 1.05,
                "decay_rate": 0.04,
                "max_years": 6,
                "institutions": {}
            }
        }
        
        # Add detailed information for each college
        for tier, colleges in [("tier_1", tier_1_colleges), ("tier_2", tier_2_colleges), ("tier_3", tier_3_colleges)]:
            for college in colleges:
                college_key = college.lower().replace(" ", "_").replace(",", "")
                premium_colleges[tier]["institutions"][college_key] = {
                    "name": college,
                    "established": random.randint(1950, 2010),
                    "location": random.choice(["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"]),
                    "specializations": random.sample([
                        "Computer Science", "Information Technology", "Electronics", 
                        "Mechanical", "Civil", "Electrical", "Chemical", "Aerospace"
                    ], random.randint(3, 6)),
                    "placement_rate": random.uniform(0.75, 0.95) if tier == "tier_1" else random.uniform(0.60, 0.85),
                    "average_package": random.randint(800000, 2000000) if tier == "tier_1" else random.randint(400000, 1200000)
                }
        
        return premium_colleges
    
    def generate_salary_records(self, num_records: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate realistic salary records for different experience levels and backgrounds
        
        Args:
            num_records (int): Number of salary records to generate
            
        Returns:
            List[Dict[str, Any]]: List of salary records
        """
        salary_records = []
        experience_levels = ["freshers", "junior", "mid_level", "senior"]
        locations = ["bangalore", "mumbai", "delhi", "hyderabad", "pune", "chennai", "kolkata"]
        roles = ["software_engineer", "senior_developer", "technical_lead", "project_manager", "data_scientist", "devops_engineer"]
        
        for i in range(num_records):
            # Random selections
            experience_level = random.choice(experience_levels)
            location = random.choice(locations)
            role = random.choice(roles)
            
            # Experience in years based on level
            if experience_level == "freshers":
                years_experience = random.uniform(0, 1)
            elif experience_level == "junior":
                years_experience = random.uniform(1, 3)
            elif experience_level == "mid_level":
                years_experience = random.uniform(3, 7)
            else:  # senior
                years_experience = random.uniform(7, 15)
            
            # College background (30% from premium colleges)
            has_premium_college = random.random() < 0.3
            college_tier = None
            college_name = None
            
            if has_premium_college:
                college_tier = random.choices(
                    ["tier_1", "tier_2", "tier_3"],
                    weights=[0.2, 0.4, 0.4]
                )[0]
                tier_colleges = list(self.synthetic_data["premium_colleges"][college_tier]["institutions"].keys())
                if tier_colleges:
                    college_name = random.choice(tier_colleges)
            
            # Base salary calculation
            if "salary_bands" in self.config and experience_level in self.config["salary_bands"]:
                if "base" in self.config["salary_bands"][experience_level]:
                    base_salary = self.config["salary_bands"][experience_level]["base"]
                else:
                    # Handle different config structure
                    band_config = self.config["salary_bands"][experience_level]
                    if "min" in band_config and "max" in band_config:
                        base_salary = (band_config["min"] + band_config["max"]) // 2
                    else:
                        base_salary = 600000  # Default fallback
            else:
                # Fallback salary bands
                salary_defaults = {
                    "freshers": 400000,
                    "junior": 800000,
                    "mid_level": 1500000,
                    "senior": 2500000
                }
                base_salary = salary_defaults.get(experience_level, 600000)
            
            # Apply variations
            salary_variation = random.uniform(0.8, 1.2)  # ±20% variation
            calculated_salary = int(base_salary * salary_variation)
            
            # Apply college premium
            if has_premium_college and college_tier:
                college_multiplier = self.synthetic_data["premium_colleges"][college_tier]["multiplier"]
                # Apply decay based on years of experience
                decay_rate = self.synthetic_data["premium_colleges"][college_tier]["decay_rate"]
                effective_multiplier = max(1.0, college_multiplier - (years_experience * decay_rate))
                calculated_salary = int(calculated_salary * effective_multiplier)
            
            # Apply location factor
            location_factors = {
                "bangalore": 1.1, "mumbai": 1.15, "delhi": 1.05,
                "hyderabad": 1.0, "pune": 0.95, "chennai": 0.9, "kolkata": 0.85
            }
            calculated_salary = int(calculated_salary * location_factors.get(location, 1.0))
            
            # Apply role multiplier
            role_multipliers = {
                "software_engineer": 1.0, "senior_developer": 1.15, "technical_lead": 1.3,
                "project_manager": 1.25, "data_scientist": 1.2, "devops_engineer": 1.1
            }
            calculated_salary = int(calculated_salary * role_multipliers.get(role, 1.0))
            
            # Generate skills (affects salary)
            skills = random.sample([
                "Python", "Java", "JavaScript", "React", "Node.js", "AWS", "Docker",
                "Kubernetes", "Machine Learning", "Data Analysis", "SQL", "MongoDB",
                "Spring Boot", "Angular", "Vue.js", "DevOps", "CI/CD", "Microservices"
            ], random.randint(3, 8))
            
            # Skill premium (high-demand skills)
            premium_skills = ["Machine Learning", "AWS", "Kubernetes", "DevOps", "Data Analysis"]
            skill_premium = sum(0.05 for skill in skills if skill in premium_skills)
            calculated_salary = int(calculated_salary * (1 + skill_premium))
            
            # Create record
            record = {
                "id": f"emp_{i+1:04d}",
                "experience_level": experience_level,
                "years_experience": round(years_experience, 1),
                "location": location,
                "role": role,
                "annual_salary": calculated_salary,
                "base_salary": base_salary,
                "college_tier": college_tier,
                "college_name": college_name,
                "skills": skills,
                "generated_at": datetime.now().isoformat(),
                "salary_components": {
                    "base": int(calculated_salary * 0.7),
                    "variable": int(calculated_salary * 0.2),
                    "benefits": int(calculated_salary * 0.1)
                }
            }
            
            salary_records.append(record)
        
        return salary_records
    
    def generate_growth_patterns(self) -> Dict[str, Any]:
        """
        Generate realistic salary growth patterns over time
        
        Returns:
            Dict[str, Any]: Growth pattern data
        """
        growth_patterns = {
            "yearly_growth": {
                "freshers": {"min": 0.15, "max": 0.25, "average": 0.20},
                "junior": {"min": 0.12, "max": 0.20, "average": 0.16},
                "mid_level": {"min": 0.10, "max": 0.18, "average": 0.14},
                "senior": {"min": 0.08, "max": 0.15, "average": 0.12}
            },
            "role_transition_multipliers": {
                "software_engineer_to_senior_developer": 1.25,
                "senior_developer_to_technical_lead": 1.30,
                "technical_lead_to_project_manager": 1.15,
                "any_to_data_scientist": 1.35,
                "any_to_devops_engineer": 1.20
            },
            "skill_acquisition_bonus": {
                "cloud_certification": 0.10,
                "machine_learning_expertise": 0.15,
                "leadership_skills": 0.12,
                "domain_expertise": 0.08
            },
            "market_trends": {
                "2024": {"inflation_factor": 1.06, "demand_multiplier": 1.1},
                "2025": {"inflation_factor": 1.05, "demand_multiplier": 1.15},
                "2026": {"inflation_factor": 1.04, "demand_multiplier": 1.12}
            }
        }
        
        return growth_patterns
    
    def generate_complete_database(self, num_salary_records: int = 1000) -> Dict[str, Any]:
        """
        Generate the complete synthetic database
        
        Args:
            num_salary_records (int): Number of salary records to generate
            
        Returns:
            Dict[str, Any]: Complete synthetic database
        """
        print("Generating OTS Synthetic Database...")
        
        # Generate all components
        print("- Generating premium college catalog...")
        self.synthetic_data["premium_colleges"] = self.generate_premium_college_catalog()
        
        print(f"- Generating {num_salary_records} salary records...")
        self.synthetic_data["salary_records"] = self.generate_salary_records(num_salary_records)
        
        print("- Generating growth patterns...")
        self.synthetic_data["growth_patterns"] = self.generate_growth_patterns()
        
        # Update metadata
        self.synthetic_data["metadata"]["total_records"] = len(self.synthetic_data["salary_records"])
        self.synthetic_data["metadata"]["premium_colleges_count"] = sum(
            len(tier["institutions"]) for tier in self.synthetic_data["premium_colleges"].values()
        )
        
        print("✓ Synthetic database generation completed!")
        return self.synthetic_data
    
    def save_database(self, output_path: str = "ots_synthetic_database.json") -> None:
        """
        Save the synthetic database to a JSON file
        
        Args:
            output_path (str): Path to save the database file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.synthetic_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Database saved to {output_path}")
            print(f"  - Total salary records: {len(self.synthetic_data['salary_records'])}")
            print(f"  - Premium colleges: {self.synthetic_data['metadata']['premium_colleges_count']}")
            print(f"  - File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the synthetic database
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        if not self.synthetic_data["salary_records"]:
            return {"error": "No data generated yet"}
        
        salaries = [record["annual_salary"] for record in self.synthetic_data["salary_records"]]
        
        stats = {
            "total_records": len(self.synthetic_data["salary_records"]),
            "salary_statistics": {
                "min": min(salaries),
                "max": max(salaries),
                "average": int(sum(salaries) / len(salaries)),
                "median": sorted(salaries)[len(salaries) // 2]
            },
            "experience_distribution": {},
            "location_distribution": {},
            "college_tier_distribution": {}
        }
        
        # Calculate distributions
        for record in self.synthetic_data["salary_records"]:
            # Experience level distribution
            exp_level = record["experience_level"]
            stats["experience_distribution"][exp_level] = stats["experience_distribution"].get(exp_level, 0) + 1
            
            # Location distribution
            location = record["location"]
            stats["location_distribution"][location] = stats["location_distribution"].get(location, 0) + 1
            
            # College tier distribution
            college_tier = record["college_tier"] or "no_premium_college"
            stats["college_tier_distribution"][college_tier] = stats["college_tier_distribution"].get(college_tier, 0) + 1
        
        return stats


def main():
    """
    Main function to generate and save the OTS synthetic database
    """
    print("=" * 60)
    print("OTS Synthetic Database Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = OTSSyntheticDatabase()
    
    # Generate complete database
    database = generator.generate_complete_database(num_salary_records=1500)
    
    # Save to file
    generator.save_database("ots_synthetic_database.json")
    
    # Display statistics
    print("\n" + "=" * 40)
    print("DATABASE STATISTICS")
    print("=" * 40)
    stats = generator.get_statistics()
    
    print(f"Total Records: {stats['total_records']}")
    print(f"Salary Range: ₹{stats['salary_statistics']['min']:,} - ₹{stats['salary_statistics']['max']:,}")
    print(f"Average Salary: ₹{stats['salary_statistics']['average']:,}")
    
    print("\nExperience Level Distribution:")
    for level, count in stats["experience_distribution"].items():
        percentage = (count / stats['total_records']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    print("\nCollege Tier Distribution:")
    for tier, count in stats["college_tier_distribution"].items():
        percentage = (count / stats['total_records']) * 100
        print(f"  {tier}: {count} ({percentage:.1f}%)")
    
    print("\n✓ OTS Synthetic Database generation completed successfully!")


if __name__ == "__main__":
    main()