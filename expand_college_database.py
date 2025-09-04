#!/usr/bin/env python3
"""
College Database Expansion Script
Expands the OTS synthetic database with comprehensive Tier 2 and Tier 3 colleges across India

This script adds:
- 50+ additional Tier 2 colleges (NITs, IIITs, State Universities, Premier Private Colleges)
- 100+ additional Tier 3 colleges (Regional Universities, Private Colleges, Deemed Universities)
- Proper location mapping for city-based salary calculations
- Realistic placement rates and package data

Author: OTS Development Team
Version: 1.0.0
Date: 2025-01-15
"""

import json
import random
from typing import Dict, List, Any
from datetime import datetime

class CollegeDatabaseExpander:
    """
    Expands the college database with comprehensive Tier 2 and Tier 3 institutions
    """
    
    def __init__(self, database_path: str = "ots_synthetic_database.json"):
        """
        Initialize the database expander
        
        Args:
            database_path (str): Path to the OTS synthetic database
        """
        self.database_path = database_path
        self.database = {}
        self.load_database()
        
        # Define specializations for realistic data
        self.specializations = [
            "Computer Science", "Information Technology", "Electronics", "Electrical",
            "Mechanical", "Civil", "Chemical", "Aerospace", "Biotechnology",
            "Data Science", "Artificial Intelligence", "Cybersecurity", "Software Engineering"
        ]
        
        # Define major Indian cities for location mapping
        self.major_cities = [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune",
            "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
            "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara", "Ghaziabad",
            "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan-Dombivali",
            "Vasai-Virar", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar",
            "Navi Mumbai", "Allahabad", "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior"
        ]
    
    def load_database(self) -> None:
        """
        Load the existing database
        """
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
        except FileNotFoundError:
            print(f"Database file not found: {self.database_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in database: {e}")
            return
    
    def normalize_college_name(self, name: str) -> str:
        """
        Normalize college name for database key
        
        Args:
            name (str): College name
            
        Returns:
            str: Normalized key
        """
        import re
        normalized = re.sub(r'[^a-z0-9\s]', '', name.lower())
        return re.sub(r'\s+', '_', normalized.strip())
    
    def generate_college_data(self, name: str, tier: str, location: str = None) -> Dict[str, Any]:
        """
        Generate realistic college data
        
        Args:
            name (str): College name
            tier (str): College tier (tier_2 or tier_3)
            location (str): College location
            
        Returns:
            Dict[str, Any]: College data
        """
        if not location:
            location = random.choice(self.major_cities)
        
        # Generate realistic placement rates and packages based on tier
        if tier == "tier_2":
            placement_rate = round(random.uniform(0.65, 0.85), 4)
            avg_package = random.randint(500000, 1200000)
        else:  # tier_3
            placement_rate = round(random.uniform(0.45, 0.75), 4)
            avg_package = random.randint(300000, 800000)
        
        # Select random specializations
        num_specializations = random.randint(3, 6)
        specializations = random.sample(self.specializations, num_specializations)
        
        # Generate establishment year
        established = random.randint(1950, 2010)
        
        return {
            "name": name,
            "established": established,
            "location": location,
            "specializations": specializations,
            "placement_rate": placement_rate,
            "average_package": avg_package
        }
    
    def get_tier_2_colleges(self) -> List[Dict[str, str]]:
        """
        Get comprehensive list of Tier 2 colleges
        
        Returns:
            List[Dict[str, str]]: List of college names and locations
        """
        tier_2_colleges = [
            # Additional NITs
            {"name": "National Institute of Technology Trichy", "location": "Tiruchirappalli"},
            {"name": "National Institute of Technology Warangal", "location": "Warangal"},
            {"name": "National Institute of Technology Rourkela", "location": "Rourkela"},
            {"name": "National Institute of Technology Kurukshetra", "location": "Kurukshetra"},
            {"name": "National Institute of Technology Durgapur", "location": "Durgapur"},
            {"name": "National Institute of Technology Jamshedpur", "location": "Jamshedpur"},
            {"name": "National Institute of Technology Hamirpur", "location": "Hamirpur"},
            {"name": "National Institute of Technology Jalandhar", "location": "Jalandhar"},
            {"name": "National Institute of Technology Patna", "location": "Patna"},
            {"name": "National Institute of Technology Raipur", "location": "Raipur"},
            {"name": "National Institute of Technology Silchar", "location": "Silchar"},
            {"name": "National Institute of Technology Agartala", "location": "Agartala"},
            {"name": "National Institute of Technology Arunachal Pradesh", "location": "Yupia"},
            {"name": "National Institute of Technology Delhi", "location": "Delhi"},
            {"name": "National Institute of Technology Goa", "location": "Goa"},
            {"name": "National Institute of Technology Manipur", "location": "Imphal"},
            {"name": "National Institute of Technology Meghalaya", "location": "Shillong"},
            {"name": "National Institute of Technology Mizoram", "location": "Aizawl"},
            {"name": "National Institute of Technology Nagaland", "location": "Dimapur"},
            {"name": "National Institute of Technology Puducherry", "location": "Puducherry"},
            {"name": "National Institute of Technology Sikkim", "location": "Ravangla"},
            {"name": "National Institute of Technology Uttarakhand", "location": "Srinagar"},
            
            # Additional IIITs
            {"name": "Indian Institute of Information Technology Gwalior", "location": "Gwalior"},
            {"name": "Indian Institute of Information Technology Jabalpur", "location": "Jabalpur"},
            {"name": "Indian Institute of Information Technology Kancheepuram", "location": "Chennai"},
            {"name": "Indian Institute of Information Technology Lucknow", "location": "Lucknow"},
            {"name": "Indian Institute of Information Technology Pune", "location": "Pune"},
            {"name": "Indian Institute of Information Technology Vadodara", "location": "Vadodara"},
            {"name": "Indian Institute of Information Technology Nagpur", "location": "Nagpur"},
            {"name": "Indian Institute of Information Technology Kota", "location": "Kota"},
            {"name": "Indian Institute of Information Technology Sonepat", "location": "Sonepat"},
            {"name": "Indian Institute of Information Technology Una", "location": "Una"},
            {"name": "Indian Institute of Information Technology Surat", "location": "Surat"},
            {"name": "Indian Institute of Information Technology Bhopal", "location": "Bhopal"},
            {"name": "Indian Institute of Information Technology Bhagalpur", "location": "Bhagalpur"},
            {"name": "Indian Institute of Information Technology Agartala", "location": "Agartala"},
            {"name": "Indian Institute of Information Technology Kalyani", "location": "Kalyani"},
            {"name": "Indian Institute of Information Technology Ranchi", "location": "Ranchi"},
            {"name": "Indian Institute of Information Technology Manipur", "location": "Imphal"},
            {"name": "Indian Institute of Information Technology Kurnool", "location": "Kurnool"},
            {"name": "Indian Institute of Information Technology Tiruchirappalli", "location": "Tiruchirappalli"},
            {"name": "Indian Institute of Information Technology Dharwad", "location": "Dharwad"},
            
            # State Universities and Premier Colleges
            {"name": "Delhi Technological University", "location": "Delhi"},
            {"name": "Netaji Subhas University of Technology", "location": "Delhi"},
            {"name": "Birla Institute of Technology and Science Pilani", "location": "Pilani"},
            {"name": "Birla Institute of Technology and Science Goa", "location": "Goa"},
            {"name": "Birla Institute of Technology and Science Hyderabad", "location": "Hyderabad"},
            {"name": "Vellore Institute of Technology", "location": "Vellore"},
            {"name": "Vellore Institute of Technology Chennai", "location": "Chennai"},
            {"name": "Vellore Institute of Technology Bhopal", "location": "Bhopal"},
            {"name": "Manipal Institute of Technology", "location": "Manipal"},
            {"name": "International Institute of Information Technology Hyderabad", "location": "Hyderabad"},
            {"name": "International Institute of Information Technology Bangalore", "location": "Bangalore"},
            {"name": "Indraprastha Institute of Information Technology Delhi", "location": "Delhi"},
            {"name": "Thapar Institute of Engineering and Technology", "location": "Patiala"},
            {"name": "Bharati Vidyapeeth College of Engineering", "location": "Pune"},
            {"name": "Coimbatore Institute of Technology", "location": "Coimbatore"},
            {"name": "Government College of Technology Coimbatore", "location": "Coimbatore"},
            {"name": "Kumaraguru College of Technology", "location": "Coimbatore"},
            {"name": "SSN College of Engineering", "location": "Chennai"},
            {"name": "Ramaiah Institute of Technology", "location": "Bangalore"},
            {"name": "BMS College of Engineering", "location": "Bangalore"},
            {"name": "RV College of Engineering", "location": "Bangalore"},
            {"name": "PES University", "location": "Bangalore"},
            {"name": "Dayananda Sagar College of Engineering", "location": "Bangalore"},
            {"name": "Birla Institute of Technology Mesra", "location": "Ranchi"},
            {"name": "Jamia Millia Islamia", "location": "Delhi"},
            {"name": "Aligarh Muslim University", "location": "Aligarh"},
            {"name": "Banaras Hindu University", "location": "Varanasi"},
            {"name": "University of Hyderabad", "location": "Hyderabad"},
            {"name": "Osmania University", "location": "Hyderabad"},
            {"name": "Jawaharlal Nehru Technological University Hyderabad", "location": "Hyderabad"},
            {"name": "Jawaharlal Nehru Technological University Kakinada", "location": "Kakinada"},
            {"name": "Jawaharlal Nehru Technological University Anantapur", "location": "Anantapur"},
            {"name": "Andhra University", "location": "Visakhapatnam"},
            {"name": "National Institute of Technology Karnataka", "location": "Surathkal"},
            {"name": "Bangalore Institute of Technology", "location": "Bangalore"},
            {"name": "Malnad College of Engineering", "location": "Hassan"},
            {"name": "Siddaganga Institute of Technology", "location": "Tumkur"},
            {"name": "Walchand College of Engineering", "location": "Sangli"},
            {"name": "Government College of Engineering Pune", "location": "Pune"},
            {"name": "Veermata Jijabai Technological Institute", "location": "Mumbai"},
            {"name": "Institute of Chemical Technology", "location": "Mumbai"},
            {"name": "Sardar Patel Institute of Technology", "location": "Mumbai"},
            {"name": "K J Somaiya College of Engineering", "location": "Mumbai"},
            {"name": "Fr Conceicao Rodrigues College of Engineering", "location": "Mumbai"},
            {"name": "Thadomal Shahani Engineering College", "location": "Mumbai"},
            {"name": "DJ Sanghvi College of Engineering", "location": "Mumbai"},
            {"name": "Mukesh Patel School of Technology Management and Engineering", "location": "Mumbai"},
            {"name": "Dwarkadas J Sanghvi College of Engineering", "location": "Mumbai"},
            {"name": "Atharva College of Engineering", "location": "Mumbai"},
            {"name": "Shri Ramdeobaba College of Engineering and Management", "location": "Nagpur"},
            {"name": "Visvesvaraya National Institute of Technology", "location": "Nagpur"},
            {"name": "Government College of Engineering Nagpur", "location": "Nagpur"},
            {"name": "Yeshwantrao Chavan College of Engineering", "location": "Nagpur"},
            {"name": "Priyadarshini College of Engineering", "location": "Nagpur"},
            {"name": "Shri Guru Gobind Singhji Institute of Engineering and Technology", "location": "Nanded"},
            {"name": "Government College of Engineering Aurangabad", "location": "Aurangabad"},
            {"name": "Maharashtra Institute of Technology", "location": "Pune"},
            {"name": "Vishwakarma Institute of Technology", "location": "Pune"},
            {"name": "Cummins College of Engineering for Women", "location": "Pune"},
            {"name": "Sinhgad College of Engineering", "location": "Pune"},
            {"name": "PCCOE Pune", "location": "Pune"},
            {"name": "Army Institute of Technology", "location": "Pune"},
            {"name": "Dr D Y Patil Institute of Technology", "location": "Pune"},
            {"name": "Zeal College of Engineering and Research", "location": "Pune"},
            {"name": "JSPM Narhe Technical Campus", "location": "Pune"},
            {"name": "Symbiosis Institute of Technology", "location": "Pune"}
        ]
        
        return tier_2_colleges
    
    def get_tier_3_colleges(self) -> List[Dict[str, str]]:
        """
        Get comprehensive list of Tier 3 colleges
        
        Returns:
            List[Dict[str, str]]: List of college names and locations
        """
        tier_3_colleges = [
            # Private Universities and Deemed Universities
            {"name": "Amity University Noida", "location": "Noida"},
            {"name": "Amity University Gurgaon", "location": "Gurgaon"},
            {"name": "Amity University Lucknow", "location": "Lucknow"},
            {"name": "Amity University Jaipur", "location": "Jaipur"},
            {"name": "Amity University Mumbai", "location": "Mumbai"},
            {"name": "Lovely Professional University", "location": "Phagwara"},
            {"name": "Chandigarh University", "location": "Chandigarh"},
            {"name": "Chitkara University", "location": "Chandigarh"},
            {"name": "Sharda University", "location": "Greater Noida"},
            {"name": "Bennett University", "location": "Greater Noida"},
            {"name": "Galgotias University", "location": "Greater Noida"},
            {"name": "GL Bajaj Institute of Technology and Management", "location": "Greater Noida"},
            {"name": "ABES Engineering College", "location": "Ghaziabad"},
            {"name": "KIET Group of Institutions", "location": "Ghaziabad"},
            {"name": "IMS Engineering College", "location": "Ghaziabad"},
            {"name": "Ajay Kumar Garg Engineering College", "location": "Ghaziabad"},
            {"name": "Krishna Engineering College", "location": "Ghaziabad"},
            {"name": "RD Engineering College", "location": "Ghaziabad"},
            {"name": "ABESIT", "location": "Ghaziabad"},
            {"name": "ITS Engineering College", "location": "Greater Noida"},
            {"name": "Galgotias College of Engineering and Technology", "location": "Greater Noida"},
            {"name": "GNIOT", "location": "Greater Noida"},
            {"name": "Greater Noida Institute of Technology", "location": "Greater Noida"},
            {"name": "Accurate Institute of Management and Technology", "location": "Greater Noida"},
            {"name": "Lloyd Institute of Engineering and Technology", "location": "Greater Noida"},
            {"name": "Noida Institute of Engineering and Technology", "location": "Greater Noida"},
            {"name": "JSS Academy of Technical Education", "location": "Noida"},
            {"name": "Dronacharya College of Engineering", "location": "Gurgaon"},
            {"name": "ITM University", "location": "Gurgaon"},
            {"name": "Ansal University", "location": "Gurgaon"},
            {"name": "The NorthCap University", "location": "Gurgaon"},
            {"name": "Manav Rachna International Institute of Research and Studies", "location": "Faridabad"},
            {"name": "Lingayas Vidyapeeth", "location": "Faridabad"},
            {"name": "YMC University of Science and Technology", "location": "Faridabad"},
            {"name": "SGT University", "location": "Gurgaon"},
            {"name": "GD Goenka University", "location": "Gurgaon"},
            {"name": "Maharishi Markandeshwar University", "location": "Ambala"},
            {"name": "Kurukshetra University", "location": "Kurukshetra"},
            {"name": "Deenbandhu Chhotu Ram University of Science and Technology", "location": "Murthal"},
            {"name": "Maharishi Dayanand University", "location": "Rohtak"},
            {"name": "Chaudhary Devi Lal University", "location": "Sirsa"},
            {"name": "Guru Jambheshwar University of Science and Technology", "location": "Hisar"},
            {"name": "Pandit Deendayal Petroleum University", "location": "Gandhinagar"},
            {"name": "Nirma University", "location": "Ahmedabad"},
            {"name": "Charotar University of Science and Technology", "location": "Anand"},
            {"name": "Ganpat University", "location": "Mehsana"},
            {"name": "Gujarat Technological University", "location": "Ahmedabad"},
            {"name": "Marwadi University", "location": "Rajkot"},
            {"name": "RK University", "location": "Rajkot"},
            {"name": "Parul University", "location": "Vadodara"},
            {"name": "Unitedworld Institute of Technology", "location": "Gandhinagar"},
            {"name": "Institute of Technology Nirma University", "location": "Ahmedabad"},
            {"name": "LD College of Engineering", "location": "Ahmedabad"},
            {"name": "Government Engineering College Gandhinagar", "location": "Gandhinagar"},
            {"name": "Government Engineering College Rajkot", "location": "Rajkot"},
            {"name": "Government Engineering College Surat", "location": "Surat"},
            {"name": "Government Engineering College Vadodara", "location": "Vadodara"},
            {"name": "Sarvajanik College of Engineering and Technology", "location": "Surat"},
            {"name": "Shantilal Shah Engineering College", "location": "Bhavnagar"},
            {"name": "Vishwakarma Government Engineering College", "location": "Ahmedabad"},
            {"name": "Kalinga Institute of Industrial Technology", "location": "Bhubaneswar"},
            {"name": "Siksha O Anusandhan University", "location": "Bhubaneswar"},
            {"name": "Institute of Technical Education and Research", "location": "Bhubaneswar"},
            {"name": "Centurion University of Technology and Management", "location": "Bhubaneswar"},
            {"name": "Trident Academy of Technology", "location": "Bhubaneswar"},
            {"name": "College of Engineering and Technology Bhubaneswar", "location": "Bhubaneswar"},
            {"name": "Gandhi Institute for Technology", "location": "Bhubaneswar"},
            {"name": "Indira Gandhi Institute of Technology", "location": "Sarang"},
            {"name": "Veer Surendra Sai University of Technology", "location": "Burla"},
            {"name": "National Institute of Science and Technology", "location": "Berhampur"},
            {"name": "CV Raman College of Engineering", "location": "Bhubaneswar"},
            {"name": "Silicon Institute of Technology", "location": "Bhubaneswar"},
            {"name": "Aryan Institute of Engineering and Technology", "location": "Bhubaneswar"},
            {"name": "Capital Engineering College", "location": "Bhubaneswar"},
            {"name": "Synergy Institute of Engineering and Technology", "location": "Dhenkanal"},
            {"name": "Raajdhani Engineering College", "location": "Bhubaneswar"},
            {"name": "Orissa Engineering College", "location": "Bhubaneswar"},
            {"name": "Prasad V Potluri Siddhartha Institute of Technology", "location": "Vijayawada"},
            {"name": "Vignan's Foundation for Science Technology and Research", "location": "Guntur"},
            {"name": "KL University", "location": "Guntur"},
            {"name": "VIT AP University", "location": "Amaravati"},
            {"name": "SRM University AP", "location": "Amaravati"},
            {"name": "Centurion University", "location": "Vizianagaram"},
            {"name": "Gitam University", "location": "Visakhapatnam"},
            {"name": "Anil Neerukonda Institute of Technology and Sciences", "location": "Visakhapatnam"},
            {"name": "Gayatri Vidya Parishad College of Engineering", "location": "Visakhapatnam"},
            {"name": "Dhanekula Institute of Engineering and Technology", "location": "Vijayawada"},
            {"name": "RVR and JC College of Engineering", "location": "Guntur"},
            {"name": "Bapatla Engineering College", "location": "Bapatla"},
            {"name": "Gudlavalleru Engineering College", "location": "Gudlavalleru"},
            {"name": "Kakatiya Institute of Technology and Science", "location": "Warangal"},
            {"name": "Chaitanya Bharathi Institute of Technology", "location": "Hyderabad"},
            {"name": "Gokaraju Rangaraju Institute of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Vasavi College of Engineering", "location": "Hyderabad"},
            {"name": "CVR College of Engineering", "location": "Hyderabad"},
            {"name": "Malla Reddy College of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Institute of Aeronautical Engineering", "location": "Hyderabad"},
            {"name": "Sreenidhi Institute of Science and Technology", "location": "Hyderabad"},
            {"name": "VNR Vignana Jyothi Institute of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Mahindra University", "location": "Hyderabad"},
            {"name": "ICFAI Foundation for Higher Education", "location": "Hyderabad"},
            {"name": "CMR Institute of Technology", "location": "Hyderabad"},
            {"name": "Matrusri Engineering College", "location": "Hyderabad"},
            {"name": "TKR College of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Sreyas Institute of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Anurag Group of Institutions", "location": "Hyderabad"},
            {"name": "Guru Nanak Institute of Technology", "location": "Hyderabad"},
            {"name": "Methodist College of Engineering and Technology", "location": "Hyderabad"},
            {"name": "Nalla Malla Reddy Engineering College", "location": "Hyderabad"},
            {"name": "St Martins Engineering College", "location": "Hyderabad"},
            {"name": "Jyothishmathi Institute of Technology and Science", "location": "Hyderabad"}
        ]
        
        return tier_3_colleges
    
    def expand_database(self) -> None:
        """
        Expand the database with new colleges
        """
        if "premium_colleges" not in self.database:
            print("Premium colleges section not found in database")
            return
        
        # Get new colleges
        tier_2_colleges = self.get_tier_2_colleges()
        tier_3_colleges = self.get_tier_3_colleges()
        
        # Add Tier 2 colleges
        print(f"Adding {len(tier_2_colleges)} Tier 2 colleges...")
        for college_info in tier_2_colleges:
            college_key = self.normalize_college_name(college_info["name"])
            if college_key not in self.database["premium_colleges"]["tier_2"]["institutions"]:
                college_data = self.generate_college_data(
                    college_info["name"], 
                    "tier_2", 
                    college_info["location"]
                )
                self.database["premium_colleges"]["tier_2"]["institutions"][college_key] = college_data
        
        # Add Tier 3 colleges
        print(f"Adding {len(tier_3_colleges)} Tier 3 colleges...")
        for college_info in tier_3_colleges:
            college_key = self.normalize_college_name(college_info["name"])
            if college_key not in self.database["premium_colleges"]["tier_3"]["institutions"]:
                college_data = self.generate_college_data(
                    college_info["name"], 
                    "tier_3", 
                    college_info["location"]
                )
                self.database["premium_colleges"]["tier_3"]["institutions"][college_key] = college_data
        
        # Update metadata
        if "metadata" in self.database:
            tier_2_count = len(self.database["premium_colleges"]["tier_2"]["institutions"])
            tier_3_count = len(self.database["premium_colleges"]["tier_3"]["institutions"])
            tier_1_count = len(self.database["premium_colleges"]["tier_1"]["institutions"])
            
            self.database["metadata"]["premium_colleges_count"] = tier_1_count + tier_2_count + tier_3_count
            self.database["metadata"]["last_updated"] = datetime.now().isoformat()
            
            print(f"Database updated with:")
            print(f"  Tier 1: {tier_1_count} colleges")
            print(f"  Tier 2: {tier_2_count} colleges")
            print(f"  Tier 3: {tier_3_count} colleges")
            print(f"  Total: {self.database['metadata']['premium_colleges_count']} colleges")
    
    def save_database(self) -> None:
        """
        Save the expanded database
        """
        try:
            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2, ensure_ascii=False)
            print(f"Database saved successfully to {self.database_path}")
        except Exception as e:
            print(f"Error saving database: {e}")

def main():
    """
    Main function to expand the college database
    """
    print("Starting College Database Expansion...")
    
    expander = CollegeDatabaseExpander()
    expander.expand_database()
    expander.save_database()
    
    print("College Database Expansion completed successfully!")

if __name__ == "__main__":
    main()