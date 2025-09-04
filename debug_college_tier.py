#!/usr/bin/env python3
"""
Debug script to test college tier detection in salary calculator
"""

from salary_calculator import OTSSalaryCalculator

def test_college_tier():
    """Test college tier detection with sample resume"""
    calc = OTSSalaryCalculator()
    
    # Test resume with IIT Delhi
    resume_text = """John Doe
Software Engineer
Education: Indian Institute of Technology, Delhi (2018)
Experience: 5 years at Google
Skills: Python, JavaScript, React"""
    
    print("Testing college tier detection...")
    print(f"Resume text: {resume_text}")
    print("\n" + "="*50)
    
    result = calc.calculate_salary(resume_text)
    
    print(f"College tier: {result.college_tier}")
    print(f"College name: {result.breakdown.get('college_name')}")
    print(f"College multiplier: {result.college_multiplier}")
    print(f"Premium college info: {result.breakdown.get('premium_college_info', {})}")
    print(f"Final salary: {result.final_salary}")
    print(f"Base salary: {result.base_salary}")
    
    # Test college extraction directly
    print("\n" + "="*50)
    print("Testing college extraction directly...")
    college_name, year = calc.extract_college_info(resume_text)
    print(f"Extracted college: {college_name}")
    print(f"Extracted year: {year}")
    
    # Test premium college calculation
    if college_name:
        print("\n" + "="*50)
        print("Testing premium college calculation...")
        multiplier, college_info = calc._calculate_college_premium_enhanced(college_name, 5)
        print(f"Enhanced multiplier: {multiplier}")
        print(f"Enhanced college info: {college_info}")

if __name__ == "__main__":
    test_college_tier()