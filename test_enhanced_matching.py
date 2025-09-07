#!/usr/bin/env python3
"""
Test script for enhanced job matching system with sentence transformers.
This script tests the new semantic similarity features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from job_matching_system import AdvancedJobMatcher
from gemini_semantic_extractor import ExtractedSkills, JobRequirements

def test_enhanced_matching():
    """
    Test the enhanced job matching system with sentence transformers.
    """
    print("Testing Enhanced Job Matching System...")
    
    # Initialize the matcher
    try:
        matcher = AdvancedJobMatcher()
        print("✓ AdvancedJobMatcher initialized successfully")
        
        # Check if sentence transformer is loaded
        if hasattr(matcher, 'sentence_model') and matcher.sentence_model is not None:
            print("✓ SentenceTransformer model loaded successfully")
        else:
            print("⚠ Using fallback similarity calculation (difflib)")
            
    except Exception as e:
        print(f"✗ Error initializing matcher: {e}")
        return False
    
    # Test skill similarity calculation
    try:
        # Test individual skill similarity
        similarity = matcher._calculate_skill_similarity("Python", "Python programming")
        print(f"✓ Skill similarity test: 'Python' vs 'Python programming' = {similarity:.3f}")
        
        # Test batch similarity calculation
        resume_skills = ["Python", "JavaScript", "React", "Node.js"]
        job_skills = ["Python programming", "Frontend development", "Backend APIs"]
        
        batch_result = matcher._calculate_batch_skill_similarity(resume_skills, job_skills)
        print(f"✓ Batch similarity calculation completed")
        print(f"  Matrix shape: {batch_result.shape if hasattr(batch_result, 'shape') else 'N/A'}")
        
    except Exception as e:
        print(f"✗ Error in similarity calculation: {e}")
        return False
    
    # Test detailed skill matching
    try:
        resume_skills = ["Python", "Django", "PostgreSQL", "Git"]
        job_skills = ["Python programming", "Web frameworks", "Database management"]
        
        detailed_match = matcher.get_detailed_skill_matching(resume_skills, job_skills)
        print(f"✓ Detailed skill matching completed")
        print(f"  Overall similarity: {detailed_match['overall_similarity']:.3f}")
        print(f"  Matched skills: {len(detailed_match['matched_skills'])}")
        print(f"  Missing skills: {len(detailed_match['missing_skills'])}")
        
        # Show some match details
        for match in detailed_match['matched_skills'][:2]:
            print(f"    {match['resume_skill']} → {match['job_skill']} (confidence: {match['confidence']})")
            
    except Exception as e:
        print(f"✗ Error in detailed skill matching: {e}")
        return False
    
    print("\n🎉 All tests passed! Enhanced job matching system is working correctly.")
    return True

def test_full_matching_pipeline():
    """
    Test the complete matching pipeline with sample data.
    """
    print("\nTesting Full Matching Pipeline...")
    
    try:
        matcher = AdvancedJobMatcher()
        
        # Sample resume skills
        resume_skills = ExtractedSkills(
            technical_skills=["Python", "Django", "PostgreSQL", "Docker"],
            soft_skills=["Communication", "Problem solving", "Team collaboration"],
            programming_languages=["Python", "JavaScript"],
            frameworks_tools=["Django", "React", "Git"],
            certifications=["AWS Certified Developer"],
            domains=["Web Development", "Cloud Computing"],
            confidence_score=0.85,
            extraction_timestamp=datetime.now().isoformat()
        )
        
        # Sample job requirements
        job_requirements = JobRequirements(
            required_skills=["Python programming", "Web development", "Database management"],
            preferred_skills=["Cloud platforms", "Frontend frameworks"],
            soft_skills_required=["Communication", "Teamwork"],
            experience_required="3+ years",
            job_role="Senior Developer",
            industry="Technology",
            education_requirements=["Bachelor's degree in Computer Science"],
            confidence_score=0.9
        )
        
        # Test enhanced skill analysis
        skill_analysis = matcher._analyze_skills_match(resume_skills, job_requirements)
        
        print(f"✓ Enhanced skill analysis completed")
        print(f"  Required skills coverage: {skill_analysis.get('required_skills_coverage', 0):.3f}")
        print(f"  Preferred skills coverage: {skill_analysis.get('preferred_skills_coverage', 0):.3f}")
        print(f"  Soft skills coverage: {skill_analysis.get('soft_skills_coverage', 0):.3f}")
        
        # Test confidence scores
        confidence_scores = skill_analysis.get('confidence_scores', {})
        print(f"  Confidence scores: {confidence_scores}")
        
        print("\n🎉 Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Enhanced Job Matching System Test Suite")
    print("=" * 50)
    
    success = True
    success &= test_enhanced_matching()
    success &= test_full_matching_pipeline()
    
    if success:
        print("\n✅ All tests passed! The enhanced job matching system is ready.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)