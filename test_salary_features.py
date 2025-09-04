#!/usr/bin/env python3
"""
Test script to demonstrate enhanced salary calculation features
including premium college integration and growth projections.
"""

from salary_calculator import OTSSalaryCalculator
import json

def test_enhanced_salary_calculation():
    """
    Test the enhanced salary calculation with premium college integration
    """
    print("=== Enhanced Salary Calculator Test ===")
    print()
    
    # Initialize calculator
    calc = OTSSalaryCalculator()
    
    # Test case 1: Premium college (IIT Delhi)
    print("Test Case 1: Premium College (IIT Delhi)")
    resume_text_iit = """
    John Doe
    Software Engineer
    
    Education:
    Bachelor of Technology, Indian Institute of Technology, Delhi (2018-2022)
    
    Experience:
    Software Engineer at Google (2022-2025) - 3 years
    - Developed scalable web applications
    - Led team of 5 developers
    """
    
    result_iit = calc.calculate_salary(resume_text_iit)
    print(f"Estimated Salary: ₹{result_iit.final_salary}L")
    print(f"College Tier: {result_iit.college_tier}")
    print(f"College Multiplier: {result_iit.college_multiplier}")
    print(f"Base Salary: ₹{result_iit.base_salary}L")
    print(f"Experience Band: {result_iit.experience_band}")
    print()
    
    # Test case 2: Non-premium college
    print("Test Case 2: Non-Premium College")
    resume_text_regular = """
    Jane Smith
    Software Developer
    
    Education:
    Bachelor of Computer Applications, Delhi University (2020-2023)
    
    Experience:
    Software Developer at TCS (2023-2025) - 2 years
    - Developed enterprise applications
    - Worked on client projects
    """
    
    result_regular = calc.calculate_salary(resume_text_regular)
    print(f"Estimated Salary: ₹{result_regular.final_salary}L")
    print(f"College Tier: {result_regular.college_tier}")
    print(f"College Multiplier: {result_regular.college_multiplier}")
    print(f"Base Salary: ₹{result_regular.base_salary}L")
    print(f"Experience Band: {result_regular.experience_band}")
    print()
    
    # Test case 3: Growth projection
    print("Test Case 3: Growth Projection (5 years)")
    growth_projection = calc.project_salary_growth(
        current_salary=result_iit.final_salary,
        years=5,
        current_experience=3,
        current_role='Software Engineer'
    )
    
    print(f"Current Salary: ₹{result_iit.final_salary}L")
    
    # Get final year projection
    final_year_key = f'year_{5}'
    if final_year_key in growth_projection['yearly_breakdown']:
        final_salary = growth_projection['yearly_breakdown'][final_year_key]['salary']
        total_growth = growth_projection['yearly_breakdown'][final_year_key]['total_growth_from_current']
        print(f"Projected Salary (5 years): ₹{final_salary:.1f}L")
        print(f"Total Growth: {total_growth:.1f}%")
    
    # Display year-wise breakdown
    print("\nYear-wise Salary Projection:")
    for year, data in growth_projection.get('yearly_breakdown', {}).items():
        print(f"  {year}: ₹{data['salary']:.1f}L (Role: {data['role']})")
    
    # Display role transitions
    if growth_projection.get('role_transitions'):
        print("\nRole Transitions:")
        for transition in growth_projection['role_transitions']:
            print(f"  Year {transition['year']}: {transition['from_role']} → {transition['to_role']} (+{transition['salary_boost']*100:.0f}%)")
    print()
    
    print("=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_enhanced_salary_calculation()