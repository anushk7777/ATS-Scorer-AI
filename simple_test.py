#!/usr/bin/env python3
"""
Simple test for the enhanced job matching system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_matching_system import AdvancedJobMatcher
from gemini_semantic_extractor import ExtractedSkills, JobRequirements

def simple_test():
    """
    Simple test of the job matching system.
    """
    print("Testing Enhanced Job Matching System...")
    
    try:
        # Initialize the matcher
        matcher = AdvancedJobMatcher()
        print("✓ AdvancedJobMatcher initialized successfully")
        
        # Test basic skill similarity
        skill1 = "Python"
        skill2 = "Python programming"
        similarity = matcher._calculate_skill_similarity(skill1, skill2)
        print(f"✓ Skill similarity test: '{skill1}' vs '{skill2}' = {similarity:.2f}")
        
        # Test with sample data
        resume_skills = ExtractedSkills(
            technical_skills=["Python", "JavaScript", "React"],
            soft_skills=["Communication", "Problem solving"],
            certifications=["AWS Certified"],
            programming_languages=["Python", "JavaScript"],
            frameworks=["React", "Django"]
        )
        
        job_requirements = JobRequirements(
            required_skills=["Python", "React", "SQL"],
            preferred_skills=["AWS", "Docker"],
            soft_skills=["Communication", "Teamwork"],
            experience_years=3,
            education_level="Bachelor's"
        )
        
        # Test detailed skill matching
        detailed_match = matcher.get_detailed_skill_matching(resume_skills, job_requirements)
        print(f"✓ Detailed skill matching completed")
        print(f"  - Required skills coverage: {detailed_match['required_skills_coverage']:.1%}")
        print(f"  - Matched required skills: {len(detailed_match['matched_required'])}")
        print(f"  - Missing required skills: {len(detailed_match['missing_required'])}")
        
        print("\n✅ All tests passed! Enhanced job matching system is working.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    simple_test()