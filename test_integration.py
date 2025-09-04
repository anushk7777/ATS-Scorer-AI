#!/usr/bin/env python3
"""
Integration Test Script for ATS Resume Analyzer

This script tests the complete pipeline from resume upload to salary prediction
and job matching functionality. It verifies that all API endpoints are working
correctly and the integration between frontend and backend is functional.

Author: AI Assistant
Date: 2025-01-27
Version: 1.0.0
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional

# Configuration
BASE_URL = os.getenv('API_URL', 'http://localhost:5000')
TEST_TIMEOUT = int(os.getenv('TEST_TIMEOUT', '30'))  # seconds

# Test data
SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer

Experience:
- 5 years of Python development
- 3 years of React and JavaScript
- Experience with AWS, Docker, and Kubernetes
- Machine Learning with TensorFlow and scikit-learn
- Database management with PostgreSQL and MongoDB

Education:
- Bachelor's in Computer Science
- Master's in Data Science

Skills:
- Programming: Python, JavaScript, Java, C++
- Web Development: React, Node.js, HTML, CSS
- Cloud: AWS, Azure, Google Cloud
- DevOps: Docker, Kubernetes, Jenkins
- Data Science: TensorFlow, PyTorch, Pandas, NumPy
"""

SAMPLE_JOB_DESCRIPTION = """
Senior Software Engineer - AI/ML

We are looking for a Senior Software Engineer with expertise in AI/ML to join our team.

Required Skills:
- 4+ years of Python development experience
- Strong background in Machine Learning and Deep Learning
- Experience with TensorFlow, PyTorch, or similar ML frameworks
- Knowledge of cloud platforms (AWS, Azure, or GCP)
- Experience with containerization (Docker, Kubernetes)
- Strong problem-solving and analytical skills

Preferred Qualifications:
- Master's degree in Computer Science, Data Science, or related field
- Experience with MLOps and model deployment
- Knowledge of distributed systems
- Experience with React or other frontend frameworks

Salary Range: $120,000 - $180,000
"""

class IntegrationTester:
    """
    Comprehensive integration tester for the ATS Resume Analyzer system.
    
    This class provides methods to test all major functionalities including:
    - Health check
    - Salary prediction
    - Job description analysis
    - Resume-job matching
    """
    
    def __init__(self, base_url: str = BASE_URL):
        """
        Initialize the integration tester.
        
        Args:
            base_url (str): Base URL of the backend server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """
        Log test results for reporting.
        
        Args:
            test_name (str): Name of the test
            success (bool): Whether the test passed
            message (str): Test result message
            details (Optional[Dict]): Additional test details
        """
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}: {message}")
        
    def test_health_check(self) -> bool:
        """
        Test the health check endpoint.
        
        Returns:
            bool: True if health check passes, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Health Check", True, "Backend server is healthy", data)
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected health status: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Connection error: {str(e)}")
            return False
    
    def test_salary_prediction(self) -> bool:
        """
        Test the salary prediction endpoint.
        
        Returns:
            bool: True if salary prediction works, False otherwise
        """
        try:
            payload = {
                "resume_text": SAMPLE_RESUME_TEXT
            }
            
            response = self.session.post(
                f"{self.base_url}/api/predict-salary",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "predicted_salary" in data and "confidence" in data:
                    salary = data["predicted_salary"]
                    confidence = data["confidence"]
                    
                    self.log_test(
                        "Salary Prediction", 
                        True, 
                        f"Predicted salary: ${salary:,.2f} (confidence: {confidence:.2f})",
                        data
                    )
                    return True
                else:
                    self.log_test("Salary Prediction", False, f"Missing required fields in response: {data}")
                    return False
            else:
                self.log_test("Salary Prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Salary Prediction", False, f"Error: {str(e)}")
            return False
    
    def test_job_analysis(self) -> bool:
        """
        Test the job description analysis endpoint.
        
        Returns:
            bool: True if job analysis works, False otherwise
        """
        try:
            payload = {
                "job_description": SAMPLE_JOB_DESCRIPTION
            }
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-job",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "required_skills" in data and "experience_requirements" in data:
                    skills_count = len(data["required_skills"])
                    
                    self.log_test(
                        "Job Analysis", 
                        True, 
                        f"Analyzed job description: {skills_count} skills identified",
                        {
                            "skills_count": skills_count,
                            "has_experience_req": bool(data["experience_requirements"])
                        }
                    )
                    return True
                else:
                    self.log_test("Job Analysis", False, f"Missing required fields in response: {data}")
                    return False
            else:
                self.log_test("Job Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Job Analysis", False, f"Error: {str(e)}")
            return False
    
    def test_job_matching(self) -> bool:
        """
        Test the job matching endpoint.
        
        Returns:
            bool: True if job matching works, False otherwise
        """
        try:
            payload = {
                "resume_text": SAMPLE_RESUME_TEXT,
                "job_description": SAMPLE_JOB_DESCRIPTION
            }
            
            response = self.session.post(
                f"{self.base_url}/api/match-job",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if "overall_match_score" in data and "skills_analysis" in data:
                    match_score = data["overall_match_score"]
                    skills_analysis = data["skills_analysis"]
                    
                    self.log_test(
                        "Job Matching", 
                        True, 
                        f"Match score: {match_score:.1f}% | Skills matched: {len(skills_analysis.get('matched_skills', []))}",
                        {
                            "match_score": match_score,
                            "matched_skills_count": len(skills_analysis.get('matched_skills', [])),
                            "missing_skills_count": len(skills_analysis.get('missing_skills', []))
                        }
                    )
                    return True
                else:
                    self.log_test("Job Matching", False, f"Missing required fields in response: {data}")
                    return False
            else:
                self.log_test("Job Matching", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Job Matching", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests and return a comprehensive report.
        
        Returns:
            Dict[str, Any]: Test results summary
        """
        print("\n" + "="*60)
        print("🧪 ATS Resume Analyzer - Integration Testing")
        print("="*60)
        print(f"Testing backend at: {self.base_url}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n")
        
        # Run all tests
        tests = [
            ("Health Check", self.test_health_check),
            ("Salary Prediction", self.test_salary_prediction),
            ("Job Analysis", self.test_job_analysis),
            ("Job Matching", self.test_job_matching)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name}...")
            if test_func():
                passed_tests += 1
            time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        success_rate = (passed_tests / total_tests) * 100
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("🎉 All tests passed! Integration is working correctly.")
        elif success_rate >= 75:
            print("⚠️  Most tests passed, but some issues detected.")
        else:
            print("❌ Multiple test failures detected. Please check the logs.")
        
        # Save detailed results
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "test_results": self.test_results
        }
        
        with open("integration_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: integration_test_report.json")
        print("="*60)
        
        return report

def main():
    """
    Main function to run integration tests.
    """
    tester = IntegrationTester()
    report = tester.run_all_tests()
    
    # Exit with appropriate code
    if report["success_rate"] == 100:
        exit(0)  # Success
    else:
        exit(1)  # Some tests failed

if __name__ == "__main__":
    main()