#!/usr/bin/env python3
"""
Premium College Database Module
Manages premium college data with tier classifications, weightage factors, and time decay calculations

This module provides:
- College tier classification (Tier 1, Tier 2, Tier 3)
- Weightage multipliers for salary calculations
- Time-based decay factors for college premium
- College search and matching functionality
- Integration with OTS Salary Calculator

Author: OTS Development Team
Version: 1.0.0
Date: 2025-01-15
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PremiumCollegeDatabase:
    """
    Manages premium college database with tier-based salary weightage calculations
    
    This class handles:
    - Loading college data from synthetic database
    - College name matching and normalization
    - Tier-based weightage calculations
    - Time decay factor applications
    - College search functionality
    """
    
    def __init__(self, database_path: str = "ots_synthetic_database.json"):
        """
        Initialize the Premium College Database
        
        Args:
            database_path (str): Path to the synthetic database JSON file
        """
        self.database_path = database_path
        self.college_data = {}
        self.tier_config = {}
        self.name_variations = {}
        self.search_index = {}
        
        # Load college data
        self._load_college_database()
        self._build_search_index()
        
        logger.info(f"Premium College Database initialized with {len(self.college_data)} institutions")
    
    def _load_college_database(self) -> None:
        """
        Load college data from the synthetic database
        
        Raises:
            FileNotFoundError: If database file is not found
            json.JSONDecodeError: If database file is invalid JSON
        """
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                database = json.load(f)
            
            # Extract premium college data
            if "premium_colleges" in database:
                self.tier_config = database["premium_colleges"]
                
                # Flatten college data for easier access
                for tier, tier_data in self.tier_config.items():
                    if "institutions" in tier_data:
                        for college_key, college_info in tier_data["institutions"].items():
                            self.college_data[college_key] = {
                                **college_info,
                                "tier": tier,
                                "multiplier": tier_data["multiplier"],
                                "decay_rate": tier_data["decay_rate"],
                                "max_years": tier_data["max_years"]
                            }
            else:
                logger.warning("No premium_colleges data found in database")
                self._create_default_college_data()
                
        except FileNotFoundError:
            logger.error(f"Database file not found: {self.database_path}")
            self._create_default_college_data()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in database file: {e}")
            self._create_default_college_data()
    
    def _create_default_college_data(self) -> None:
        """
        Create default college data if database is not available
        """
        logger.info("Creating default college data")
        
        # Default tier configuration
        self.tier_config = {
            "tier_1": {
                "multiplier": 1.30,
                "decay_rate": 0.02,
                "max_years": 10
            },
            "tier_2": {
                "multiplier": 1.15,
                "decay_rate": 0.03,
                "max_years": 8
            },
            "tier_3": {
                "multiplier": 1.05,
                "decay_rate": 0.04,
                "max_years": 6
            }
        }
        
        # Default college data (subset for demonstration)
        default_colleges = {
            "tier_1": [
                "Indian Institute of Technology Delhi",
                "Indian Institute of Technology Bombay",
                "Indian Institute of Technology Madras",
                "Indian Institute of Science Bangalore",
                "Delhi Technological University"
            ],
            "tier_2": [
                "Jadavpur University",
                "Anna University",
                "Pune Institute of Computer Technology",
                "National Institute of Technology Trichy",
                "National Institute of Technology Warangal"
            ],
            "tier_3": [
                "SRM Institute of Science and Technology",
                "Amity University",
                "Lovely Professional University",
                "Thapar Institute of Engineering and Technology",
                "Bharati Vidyapeeth College of Engineering"
            ]
        }
        
        # Create college data entries
        for tier, colleges in default_colleges.items():
            tier_data = self.tier_config[tier]
            for college_name in colleges:
                college_key = self._normalize_college_name(college_name)
                self.college_data[college_key] = {
                    "name": college_name,
                    "tier": tier,
                    "multiplier": tier_data["multiplier"],
                    "decay_rate": tier_data["decay_rate"],
                    "max_years": tier_data["max_years"],
                    "location": "India",
                    "established": 1960,
                    "specializations": ["Computer Science", "Engineering"]
                }
    
    def _normalize_college_name(self, college_name: str) -> str:
        """
        Normalize college name for consistent matching
        
        Args:
            college_name (str): Original college name
            
        Returns:
            str: Normalized college name key
        """
        if not college_name:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-z0-9\s]', '', college_name.lower())
        # Replace spaces with underscores
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized
    
    def _build_search_index(self) -> None:
        """
        Build search index for efficient college name matching
        """
        self.search_index = {}
        
        for college_key, college_data in self.college_data.items():
            college_name = college_data["name"]
            
            # Add full name
            normalized_name = self._normalize_college_name(college_name)
            self.search_index[normalized_name] = college_key
            
            # Add common abbreviations and variations
            variations = self._generate_name_variations(college_name)
            for variation in variations:
                normalized_variation = self._normalize_college_name(variation)
                if normalized_variation and normalized_variation not in self.search_index:
                    self.search_index[normalized_variation] = college_key
    
    def _generate_name_variations(self, college_name: str) -> List[str]:
        """
        Generate common variations and abbreviations for college names
        
        Args:
            college_name (str): Original college name
            
        Returns:
            List[str]: List of name variations
        """
        variations = []
        
        # Common abbreviations
        abbreviations = {
            "Indian Institute of Technology": ["IIT"],
            "National Institute of Technology": ["NIT"],
            "Indian Institute of Science": ["IISc"],
            "Indian Institute of Information Technology": ["IIIT"],
            "Delhi Technological University": ["DTU"],
            "Netaji Subhas University of Technology": ["NSUT"],
            "Birla Institute of Technology and Science": ["BITS"],
            "Vellore Institute of Technology": ["VIT"],
            "Manipal Institute of Technology": ["MIT Manipal"],
            "SRM Institute of Science and Technology": ["SRM"],
            "Lovely Professional University": ["LPU"]
        }
        
        # Add abbreviations
        for full_name, abbrevs in abbreviations.items():
            if full_name in college_name:
                for abbrev in abbrevs:
                    # Replace full name with abbreviation
                    variation = college_name.replace(full_name, abbrev)
                    variations.append(variation)
                    
                    # Add location-specific variations
                    if "IIT" in abbrev and any(city in college_name for city in ["Delhi", "Bombay", "Madras", "Kanpur"]):
                        city = next(city for city in ["Delhi", "Bombay", "Madras", "Kanpur"] if city in college_name)
                        variations.extend([f"{abbrev} {city}", f"{abbrev}-{city}"])
        
        # Add without common words
        words_to_remove = ["Institute", "University", "College", "of", "and", "Technology", "Engineering"]
        simplified = college_name
        for word in words_to_remove:
            simplified = re.sub(rf'\b{word}\b', '', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        if simplified and simplified != college_name:
            variations.append(simplified)
        
        return variations
    
    def find_college(self, college_name: str) -> Optional[Dict[str, Any]]:
        """
        Find college information by name with fuzzy matching
        
        Args:
            college_name (str): College name to search for
            
        Returns:
            Optional[Dict[str, Any]]: College information if found, None otherwise
        """
        if not college_name:
            return None
        
        # Normalize input
        normalized_input = self._normalize_college_name(college_name)
        
        # Direct match
        if normalized_input in self.search_index:
            college_key = self.search_index[normalized_input]
            return self.college_data[college_key]
        
        # Fuzzy matching - check if input contains or is contained in any college name
        for search_key, college_key in self.search_index.items():
            if (normalized_input in search_key or search_key in normalized_input) and len(normalized_input) > 3:
                return self.college_data[college_key]
        
        # Partial word matching
        input_words = set(normalized_input.split('_'))
        for search_key, college_key in self.search_index.items():
            search_words = set(search_key.split('_'))
            # If at least 2 words match and input has significant overlap
            if len(input_words.intersection(search_words)) >= 2 and len(input_words) > 1:
                return self.college_data[college_key]
        
        logger.debug(f"College not found: {college_name}")
        return None
    
    def calculate_college_weightage(self, college_name: str, years_experience: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate college weightage multiplier based on tier and experience decay
        
        Args:
            college_name (str): Name of the college
            years_experience (float): Years of professional experience
            
        Returns:
            Tuple[float, Dict[str, Any]]: (weightage_multiplier, college_info)
        """
        college_info = self.find_college(college_name)
        
        if not college_info:
            return 1.0, {"tier": "not_found", "name": college_name}
        
        # Get base multiplier and decay rate
        base_multiplier = college_info["multiplier"]
        decay_rate = college_info["decay_rate"]
        max_years = college_info["max_years"]
        
        # Apply time decay
        if years_experience <= max_years:
            # Linear decay over max_years
            decay_factor = years_experience * decay_rate
            effective_multiplier = max(1.0, base_multiplier - decay_factor)
        else:
            # After max_years, college premium is minimal
            effective_multiplier = 1.0
        
        return effective_multiplier, college_info
    
    def get_tier_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about college tiers in the database
        
        Returns:
            Dict[str, Any]: Tier statistics
        """
        tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
        tier_multipliers = {}
        
        for college_data in self.college_data.values():
            tier = college_data["tier"]
            tier_counts[tier] += 1
            tier_multipliers[tier] = college_data["multiplier"]
        
        return {
            "total_colleges": len(self.college_data),
            "tier_distribution": tier_counts,
            "tier_multipliers": tier_multipliers,
            "tier_config": self.tier_config
        }
    
    def search_colleges(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for colleges matching the query
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of matching colleges
        """
        if not query:
            return []
        
        normalized_query = self._normalize_college_name(query)
        results = []
        
        # Score-based matching
        scored_results = []
        
        for college_key, college_data in self.college_data.items():
            college_name = college_data["name"]
            normalized_name = self._normalize_college_name(college_name)
            
            score = 0
            
            # Exact match gets highest score
            if normalized_query == normalized_name:
                score = 100
            # Starts with query
            elif normalized_name.startswith(normalized_query):
                score = 80
            # Contains query
            elif normalized_query in normalized_name:
                score = 60
            # Query contains college name (for abbreviations)
            elif normalized_name in normalized_query:
                score = 50
            # Word overlap
            else:
                query_words = set(normalized_query.split('_'))
                name_words = set(normalized_name.split('_'))
                overlap = len(query_words.intersection(name_words))
                if overlap > 0:
                    score = overlap * 20
            
            if score > 0:
                scored_results.append((score, college_data))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result[1] for result in scored_results[:limit]]
    
    def get_college_by_tier(self, tier: str) -> List[Dict[str, Any]]:
        """
        Get all colleges in a specific tier
        
        Args:
            tier (str): Tier name (tier_1, tier_2, tier_3)
            
        Returns:
            List[Dict[str, Any]]: List of colleges in the tier
        """
        return [college_data for college_data in self.college_data.values() 
                if college_data["tier"] == tier]
    
    def validate_database(self) -> Dict[str, Any]:
        """
        Validate the college database for consistency
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": self.get_tier_statistics()
        }
        
        # Check for required fields
        required_fields = ["name", "tier", "multiplier", "decay_rate", "max_years"]
        
        for college_key, college_data in self.college_data.items():
            for field in required_fields:
                if field not in college_data:
                    validation_results["errors"].append(
                        f"College {college_key} missing required field: {field}"
                    )
                    validation_results["is_valid"] = False
        
        # Check tier configuration
        for tier in ["tier_1", "tier_2", "tier_3"]:
            if tier not in self.tier_config:
                validation_results["errors"].append(f"Missing tier configuration: {tier}")
                validation_results["is_valid"] = False
        
        # Check for duplicate names
        names_seen = set()
        for college_data in self.college_data.values():
            name = college_data["name"]
            if name in names_seen:
                validation_results["warnings"].append(f"Duplicate college name: {name}")
            names_seen.add(name)
        
        return validation_results


def main():
    """
    Main function for testing the Premium College Database
    """
    print("=" * 60)
    print("Premium College Database - Test Suite")
    print("=" * 60)
    
    # Initialize database
    db = PremiumCollegeDatabase()
    
    # Display statistics
    stats = db.get_tier_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Colleges: {stats['total_colleges']}")
    print(f"Tier Distribution: {stats['tier_distribution']}")
    
    # Test college search
    test_queries = [
        "IIT Delhi",
        "Indian Institute of Technology Bombay",
        "NIT Trichy",
        "SRM",
        "Amity",
        "Unknown College"
    ]
    
    print("\n" + "=" * 40)
    print("College Search Tests")
    print("=" * 40)
    
    for query in test_queries:
        college = db.find_college(query)
        if college:
            print(f"✓ '{query}' -> {college['name']} ({college['tier']})")
            
            # Test weightage calculation
            for years in [0, 2, 5, 10]:
                weightage, _ = db.calculate_college_weightage(query, years)
                print(f"  {years} years exp: {weightage:.3f}x multiplier")
        else:
            print(f"✗ '{query}' -> Not found")
        print()
    
    # Validate database
    print("=" * 40)
    print("Database Validation")
    print("=" * 40)
    validation = db.validate_database()
    print(f"Valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"Errors: {len(validation['errors'])}")
        for error in validation['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
    
    print("\n✓ Premium College Database test completed!")


if __name__ == "__main__":
    main()