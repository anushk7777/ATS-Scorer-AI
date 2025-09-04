#!/usr/bin/env python3
"""
Test script to verify the salary prediction API is working correctly.
"""

import requests
import json
import os

def test_salary_prediction():
    """
    Test the salary prediction API endpoint
    """
    # Get API URL from environment variable or use default
    api_url = os.getenv('API_URL', 'http://localhost:5000')
    url = f"{api_url}/api/predict-salary"
    
    # Sample resume text for testing
    sample_resume = """
    John Doe
    Software Engineer
    
    Education:
    B.Tech Computer Science, IIT Delhi, 2020
    
    Experience:
    Software Engineer at TCS (2020-2023) - 3 years
    - Developed web applications using Python and React
    - Worked on machine learning projects
    - Led a team of 3 developers
    
    Skills:
    Python, JavaScript, React, Django, Machine Learning, SQL
    
    Location: Bangalore
    """
    
    payload = {
        "resume_text": sample_resume
    }
    
    try:
        print("Testing salary prediction API...")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS! Salary prediction API is working.")
            print(f"Estimated Salary: ₹{result.get('estimated_salary', 'N/A')} LPA")
            print(f"Experience Level: {result.get('experience_level', 'N/A')}")
            print(f"Salary Range: ₹{result.get('salary_range', {}).get('min', 'N/A')} - ₹{result.get('salary_range', {}).get('max', 'N/A')} LPA")
            
            # Write full response to file for detailed analysis
            with open('salary_test_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("\nFull response saved to salary_test_result.json")
            return True
        else:
            print(f"\n❌ ERROR! Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR! Could not connect to the server. Is it running on port 5000?")
        return False
    except Exception as e:
        print(f"\n❌ ERROR! {str(e)}")
        return False

if __name__ == "__main__":
    test_salary_prediction()