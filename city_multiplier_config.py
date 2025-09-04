#!/usr/bin/env python3
"""
City-Based Cost of Living Multiplier Configuration
Provides location-based salary adjustments for major Indian cities

This module defines:
- City-wise cost of living multipliers (1.1x to 1.3x)
- Metropolitan area groupings
- Location detection and normalization utilities
- Integration with salary calculation system

Author: OTS Development Team
Version: 1.0.0
Date: 2025-01-15
"""

from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class CityTier(Enum):
    """
    City tier classification based on cost of living
    """
    TIER_1_METRO = "tier_1_metro"  # 1.25x - 1.3x multiplier
    TIER_1_CITY = "tier_1_city"    # 1.15x - 1.25x multiplier
    TIER_2_CITY = "tier_2_city"    # 1.1x - 1.15x multiplier
    TIER_3_CITY = "tier_3_city"    # 1.0x - 1.1x multiplier
    DEFAULT = "default"            # 1.0x multiplier

@dataclass
class CityInfo:
    """
    City information with cost of living data
    """
    name: str
    state: str
    tier: CityTier
    multiplier: float
    aliases: List[str]
    metro_area: Optional[str] = None
    population: Optional[int] = None
    tech_hub: bool = False

class CityMultiplierConfig:
    """
    Configuration class for city-based salary multipliers
    """
    
    def __init__(self):
        """
        Initialize city multiplier configuration
        """
        self.cities = self._initialize_cities()
        self.city_lookup = self._build_city_lookup()
        self.metro_areas = self._define_metro_areas()
    
    def _initialize_cities(self) -> Dict[str, CityInfo]:
        """
        Initialize comprehensive city database with cost of living multipliers
        
        Returns:
            Dict[str, CityInfo]: City database
        """
        cities = {
            # Tier 1 Metro Cities (Highest Cost of Living)
            "mumbai": CityInfo(
                name="Mumbai",
                state="Maharashtra",
                tier=CityTier.TIER_1_METRO,
                multiplier=1.30,
                aliases=["bombay", "mumbai city", "greater mumbai"],
                metro_area="Mumbai Metropolitan Region",
                population=12442373,
                tech_hub=True
            ),
            "delhi": CityInfo(
                name="Delhi",
                state="Delhi",
                tier=CityTier.TIER_1_METRO,
                multiplier=1.28,
                aliases=["new delhi", "delhi ncr", "national capital territory"],
                metro_area="National Capital Region",
                population=11034555,
                tech_hub=True
            ),
            "bangalore": CityInfo(
                name="Bangalore",
                state="Karnataka",
                tier=CityTier.TIER_1_METRO,
                multiplier=1.27,
                aliases=["bengaluru", "silicon valley of india", "garden city"],
                metro_area="Bangalore Metropolitan Area",
                population=8443675,
                tech_hub=True
            ),
            "gurgaon": CityInfo(
                name="Gurgaon",
                state="Haryana",
                tier=CityTier.TIER_1_METRO,
                multiplier=1.26,
                aliases=["gurugram", "millennium city"],
                metro_area="National Capital Region",
                population=876969,
                tech_hub=True
            ),
            "hyderabad": CityInfo(
                name="Hyderabad",
                state="Telangana",
                tier=CityTier.TIER_1_METRO,
                multiplier=1.25,
                aliases=["cyberabad", "hitec city", "city of pearls"],
                metro_area="Hyderabad Metropolitan Area",
                population=6993262,
                tech_hub=True
            ),
            
            # Tier 1 Cities (High Cost of Living)
            "pune": CityInfo(
                name="Pune",
                state="Maharashtra",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.22,
                aliases=["poona", "oxford of the east"],
                metro_area="Pune Metropolitan Region",
                population=3124458,
                tech_hub=True
            ),
            "chennai": CityInfo(
                name="Chennai",
                state="Tamil Nadu",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.21,
                aliases=["madras", "detroit of india"],
                metro_area="Chennai Metropolitan Area",
                population=4646732,
                tech_hub=True
            ),
            "kolkata": CityInfo(
                name="Kolkata",
                state="West Bengal",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.18,
                aliases=["calcutta", "city of joy"],
                metro_area="Kolkata Metropolitan Area",
                population=4496694,
                tech_hub=False
            ),
            "noida": CityInfo(
                name="Noida",
                state="Uttar Pradesh",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.20,
                aliases=["new okhla industrial development authority"],
                metro_area="National Capital Region",
                population=642381,
                tech_hub=True
            ),
            "greater_noida": CityInfo(
                name="Greater Noida",
                state="Uttar Pradesh",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.18,
                aliases=["greater noida west", "noida extension"],
                metro_area="National Capital Region",
                population=107676,
                tech_hub=True
            ),
            "faridabad": CityInfo(
                name="Faridabad",
                state="Haryana",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.17,
                aliases=[],
                metro_area="National Capital Region",
                population=1404653,
                tech_hub=False
            ),
            "ghaziabad": CityInfo(
                name="Ghaziabad",
                state="Uttar Pradesh",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.16,
                aliases=[],
                metro_area="National Capital Region",
                population=1729000,
                tech_hub=False
            ),
            "ahmedabad": CityInfo(
                name="Ahmedabad",
                state="Gujarat",
                tier=CityTier.TIER_1_CITY,
                multiplier=1.15,
                aliases=["amdavad", "manchester of india"],
                metro_area="Ahmedabad Metropolitan Area",
                population=5570585,
                tech_hub=False
            ),
            
            # Tier 2 Cities (Moderate Cost of Living)
            "kochi": CityInfo(
                name="Kochi",
                state="Kerala",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.14,
                aliases=["cochin", "queen of arabian sea"],
                population=677381,
                tech_hub=True
            ),
            "coimbatore": CityInfo(
                name="Coimbatore",
                state="Tamil Nadu",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.13,
                aliases=["kovai", "manchester of south india"],
                population=1061447,
                tech_hub=True
            ),
            "jaipur": CityInfo(
                name="Jaipur",
                state="Rajasthan",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=["pink city"],
                population=3073350,
                tech_hub=False
            ),
            "lucknow": CityInfo(
                name="Lucknow",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=["city of nawabs"],
                population=2817105,
                tech_hub=False
            ),
            "kanpur": CityInfo(
                name="Kanpur",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["manchester of the east"],
                population=2767031,
                tech_hub=False
            ),
            "nagpur": CityInfo(
                name="Nagpur",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["orange city"],
                population=2405421,
                tech_hub=False
            ),
            "indore": CityInfo(
                name="Indore",
                state="Madhya Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["commercial capital of mp"],
                population=1994397,
                tech_hub=False
            ),
            "thane": CityInfo(
                name="Thane",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.14,
                aliases=["city of lakes"],
                metro_area="Mumbai Metropolitan Region",
                population=1841488,
                tech_hub=False
            ),
            "bhopal": CityInfo(
                name="Bhopal",
                state="Madhya Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=["city of lakes"],
                population=1798218,
                tech_hub=False
            ),
            "visakhapatnam": CityInfo(
                name="Visakhapatnam",
                state="Andhra Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=["vizag", "city of destiny"],
                population=1730320,
                tech_hub=True
            ),
            "vadodara": CityInfo(
                name="Vadodara",
                state="Gujarat",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["baroda", "cultural capital of gujarat"],
                population=1670806,
                tech_hub=False
            ),
            "patna": CityInfo(
                name="Patna",
                state="Bihar",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1684222,
                tech_hub=False
            ),
            "ludhiana": CityInfo(
                name="Ludhiana",
                state="Punjab",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["manchester of india"],
                population=1618879,
                tech_hub=False
            ),
            "agra": CityInfo(
                name="Agra",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=["city of taj"],
                population=1585704,
                tech_hub=False
            ),
            "nashik": CityInfo(
                name="Nashik",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["wine capital of india"],
                population=1486973,
                tech_hub=False
            ),
            "rajkot": CityInfo(
                name="Rajkot",
                state="Gujarat",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1390933,
                tech_hub=False
            ),
            "meerut": CityInfo(
                name="Meerut",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1305429,
                tech_hub=False
            ),
            "kalyan_dombivali": CityInfo(
                name="Kalyan-Dombivali",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.13,
                aliases=["kalyan", "dombivali"],
                metro_area="Mumbai Metropolitan Region",
                population=1246381,
                tech_hub=False
            ),
            "vasai_virar": CityInfo(
                name="Vasai-Virar",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=["vasai", "virar"],
                metro_area="Mumbai Metropolitan Region",
                population=1221233,
                tech_hub=False
            ),
            "varanasi": CityInfo(
                name="Varanasi",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=["banaras", "kashi"],
                population=1198491,
                tech_hub=False
            ),
            "srinagar": CityInfo(
                name="Srinagar",
                state="Jammu and Kashmir",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=[],
                population=1192792,
                tech_hub=False
            ),
            "aurangabad": CityInfo(
                name="Aurangabad",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1175116,
                tech_hub=False
            ),
            "dhanbad": CityInfo(
                name="Dhanbad",
                state="Jharkhand",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1162472,
                tech_hub=False
            ),
            "amritsar": CityInfo(
                name="Amritsar",
                state="Punjab",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1132761,
                tech_hub=False
            ),
            "navi_mumbai": CityInfo(
                name="Navi Mumbai",
                state="Maharashtra",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.15,
                aliases=["new mumbai"],
                metro_area="Mumbai Metropolitan Region",
                population=1119477,
                tech_hub=True
            ),
            "allahabad": CityInfo(
                name="Allahabad",
                state="Uttar Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=["prayagraj"],
                population=1117094,
                tech_hub=False
            ),
            "ranchi": CityInfo(
                name="Ranchi",
                state="Jharkhand",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1073440,
                tech_hub=False
            ),
            "howrah": CityInfo(
                name="Howrah",
                state="West Bengal",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=[],
                metro_area="Kolkata Metropolitan Area",
                population=1072161,
                tech_hub=False
            ),
            "jabalpur": CityInfo(
                name="Jabalpur",
                state="Madhya Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1055525,
                tech_hub=False
            ),
            "gwalior": CityInfo(
                name="Gwalior",
                state="Madhya Pradesh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.10,
                aliases=[],
                population=1054420,
                tech_hub=False
            ),
            
            # Additional Tech Hubs
            "thiruvananthapuram": CityInfo(
                name="Thiruvananthapuram",
                state="Kerala",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.12,
                aliases=["trivandrum"],
                population=957730,
                tech_hub=True
            ),
            "mysore": CityInfo(
                name="Mysore",
                state="Karnataka",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["mysuru"],
                population=920550,
                tech_hub=True
            ),
            "mangalore": CityInfo(
                name="Mangalore",
                state="Karnataka",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["mangaluru"],
                population=623841,
                tech_hub=True
            ),
            "chandigarh": CityInfo(
                name="Chandigarh",
                state="Chandigarh",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.13,
                aliases=["city beautiful"],
                population=1055450,
                tech_hub=True
            ),
            "bhubaneswar": CityInfo(
                name="Bhubaneswar",
                state="Odisha",
                tier=CityTier.TIER_2_CITY,
                multiplier=1.11,
                aliases=["temple city"],
                population=837737,
                tech_hub=True
            )
        }
        
        return cities
    
    def _build_city_lookup(self) -> Dict[str, str]:
        """
        Build lookup table for city name normalization
        
        Returns:
            Dict[str, str]: Mapping from normalized names to city keys
        """
        lookup = {}
        
        for city_key, city_info in self.cities.items():
            # Add main name
            normalized_name = self.normalize_city_name(city_info.name)
            lookup[normalized_name] = city_key
            
            # Add aliases
            for alias in city_info.aliases:
                normalized_alias = self.normalize_city_name(alias)
                lookup[normalized_alias] = city_key
        
        return lookup
    
    def _define_metro_areas(self) -> Dict[str, List[str]]:
        """
        Define metropolitan areas and their constituent cities
        
        Returns:
            Dict[str, List[str]]: Metro area to cities mapping
        """
        metro_areas = {
            "Mumbai Metropolitan Region": ["mumbai", "thane", "kalyan_dombivali", "vasai_virar", "navi_mumbai"],
            "National Capital Region": ["delhi", "gurgaon", "noida", "greater_noida", "faridabad", "ghaziabad"],
            "Bangalore Metropolitan Area": ["bangalore"],
            "Chennai Metropolitan Area": ["chennai"],
            "Hyderabad Metropolitan Area": ["hyderabad"],
            "Pune Metropolitan Region": ["pune"],
            "Kolkata Metropolitan Area": ["kolkata", "howrah"],
            "Ahmedabad Metropolitan Area": ["ahmedabad"]
        }
        
        return metro_areas
    
    def normalize_city_name(self, city_name: str) -> str:
        """
        Normalize city name for lookup
        
        Args:
            city_name (str): Raw city name
            
        Returns:
            str: Normalized city name
        """
        if not city_name:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-z0-9\s]', '', city_name.lower())
        # Replace spaces with underscores and remove extra spaces
        normalized = re.sub(r'\s+', '_', normalized.strip())
        # Remove common suffixes
        normalized = re.sub(r'_(city|district|urban|metro|metropolitan)$', '', normalized)
        
        return normalized
    
    def get_city_multiplier(self, location: str) -> Tuple[float, Optional[CityInfo]]:
        """
        Get salary multiplier for a given location
        
        Args:
            location (str): Location string (city, state, or address)
            
        Returns:
            Tuple[float, Optional[CityInfo]]: Multiplier and city info
        """
        if not location:
            return 1.0, None
        
        # Normalize the location
        normalized_location = self.normalize_city_name(location)
        
        # Direct lookup
        if normalized_location in self.city_lookup:
            city_key = self.city_lookup[normalized_location]
            city_info = self.cities[city_key]
            return city_info.multiplier, city_info
        
        # Partial matching for compound locations
        for city_key, city_info in self.cities.items():
            city_normalized = self.normalize_city_name(city_info.name)
            if city_normalized in normalized_location or normalized_location in city_normalized:
                return city_info.multiplier, city_info
            
            # Check aliases
            for alias in city_info.aliases:
                alias_normalized = self.normalize_city_name(alias)
                if alias_normalized in normalized_location or normalized_location in alias_normalized:
                    return city_info.multiplier, city_info
        
        # Default multiplier for unknown locations
        return 1.0, None
    
    def get_metro_multiplier(self, metro_area: str) -> float:
        """
        Get average multiplier for a metropolitan area
        
        Args:
            metro_area (str): Metropolitan area name
            
        Returns:
            float: Average multiplier for the metro area
        """
        if metro_area not in self.metro_areas:
            return 1.0
        
        city_keys = self.metro_areas[metro_area]
        multipliers = [self.cities[key].multiplier for key in city_keys if key in self.cities]
        
        if not multipliers:
            return 1.0
        
        return sum(multipliers) / len(multipliers)
    
    def get_tech_hub_bonus(self, location: str) -> float:
        """
        Get additional bonus for tech hub cities
        
        Args:
            location (str): Location string
            
        Returns:
            float: Additional tech hub bonus (0.0 to 0.05)
        """
        multiplier, city_info = self.get_city_multiplier(location)
        
        if city_info and city_info.tech_hub:
            return 0.03  # 3% additional bonus for tech hubs
        
        return 0.0
    
    def detect_city_from_text(self, text: str) -> Optional[str]:
        """
        Detect city name from resume text using keyword matching
        
        Args:
            text (str): Resume text to analyze
            
        Returns:
            Optional[str]: Detected city name or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Priority order for city detection - check most specific first
        city_matches = []
        
        # Check for exact city matches and aliases
        for city_name, city_info in self.cities.items():
            # Check main city name
            if city_name.lower() in text_lower:
                city_matches.append((city_info.name, len(city_name)))
            
            # Check aliases
            for alias in city_info.aliases:
                if alias.lower() in text_lower:
                    city_matches.append((city_info.name, len(alias)))
        
        # Check lookup table for variations
        for variation, canonical_name in self.city_lookup.items():
            if variation.lower() in text_lower:
                city_matches.append((canonical_name, len(variation)))
        
        # Return the longest match (most specific)
        if city_matches:
            city_matches.sort(key=lambda x: x[1], reverse=True)
            return city_matches[0][0]
        
        return None
    
    def get_all_cities_by_tier(self, tier: CityTier) -> List[CityInfo]:
        """
        Get all cities of a specific tier
        
        Args:
            tier (CityTier): City tier
            
        Returns:
            List[CityInfo]: List of cities in the tier
        """
        return [city for city in self.cities.values() if city.tier == tier]
    
    def get_city_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the city database
        
        Returns:
            Dict[str, any]: Statistics
        """
        tier_counts = {}
        tech_hub_count = 0
        total_multiplier = 0
        
        for city in self.cities.values():
            tier_name = city.tier.value
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
            
            if city.tech_hub:
                tech_hub_count += 1
            
            total_multiplier += city.multiplier
        
        avg_multiplier = total_multiplier / len(self.cities) if self.cities else 0
        
        return {
            "total_cities": len(self.cities),
            "tier_distribution": tier_counts,
            "tech_hubs": tech_hub_count,
            "average_multiplier": round(avg_multiplier, 3),
            "multiplier_range": {
                "min": min(city.multiplier for city in self.cities.values()),
                "max": max(city.multiplier for city in self.cities.values())
            },
            "metro_areas": len(self.metro_areas)
        }

# Global instance for easy access
city_config = CityMultiplierConfig()

def get_city_multiplier(location: str) -> float:
    """
    Convenience function to get city multiplier
    
    Args:
        location (str): Location string
        
    Returns:
        float: Salary multiplier
    """
    multiplier, _ = city_config.get_city_multiplier(location)
    return multiplier

def get_location_details(location: str) -> Dict[str, any]:
    """
    Get detailed location information
    
    Args:
        location (str): Location string
        
    Returns:
        Dict[str, any]: Location details
    """
    multiplier, city_info = city_config.get_city_multiplier(location)
    tech_bonus = city_config.get_tech_hub_bonus(location)
    
    if city_info:
        return {
            "city": city_info.name,
            "state": city_info.state,
            "tier": city_info.tier.value,
            "multiplier": multiplier,
            "tech_hub": city_info.tech_hub,
            "tech_bonus": tech_bonus,
            "total_multiplier": multiplier + tech_bonus,
            "metro_area": city_info.metro_area,
            "population": city_info.population
        }
    else:
        return {
            "city": "Unknown",
            "state": "Unknown",
            "tier": "default",
            "multiplier": 1.0,
            "tech_hub": False,
            "tech_bonus": 0.0,
            "total_multiplier": 1.0,
            "metro_area": None,
            "population": None
        }

if __name__ == "__main__":
    # Test the configuration
    config = CityMultiplierConfig()
    
    print("City Multiplier Configuration Test")
    print("=" * 40)
    
    # Test some cities
    test_cities = ["Mumbai", "Bangalore", "Gurgaon", "Pune", "Chennai", "Hyderabad", "Delhi", "Unknown City"]
    
    for city in test_cities:
        details = get_location_details(city)
        print(f"{city}: {details['multiplier']:.2f}x (Tier: {details['tier']}, Tech Hub: {details['tech_hub']})")
    
    print("\nDatabase Statistics:")
    stats = config.get_city_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")