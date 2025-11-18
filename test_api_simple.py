#!/usr/bin/env python3
"""Simple test script to verify the API is working"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_screening_no_auth():
    """Test screening without authentication"""
    print("\nTesting screening without authentication...")
    
    payload = {
        "anonymized_id": "test_patient_001",
        "consent_verified": True,
        "survey_data": {
            "phq9_score": 15,
            "gad7_score": 12
        },
        "wearable_data": {
            "avg_heart_rate": 75,
            "sleep_hours": 6.5
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/screen",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Screening successful!")
            print(f"   Risk Score: {data['risk_score']['score']:.1f}")
            print(f"   Risk Level: {data['risk_score']['risk_level']}")
            print(f"   Confidence: {data['risk_score']['confidence']:.2f}")
            print(f"   Recommendations: {len(data['recommendations'])}")
            return True
        else:
            print(f"❌ Screening failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Screening failed: {e}")
        return False

def test_token_generation():
    """Test token generation endpoint"""
    print("\nTesting token generation...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/token",
            params={"user_id": "test_user", "role": "admin"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Token generation successful!")
            print(f"   Token: {data['access_token'][:50]}...")
            return data['access_token']
        else:
            print(f"❌ Token generation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Token generation failed: {e}")
        return None

def test_screening_with_auth(token):
    """Test screening with authentication"""
    print("\nTesting screening with authentication...")
    
    payload = {
        "anonymized_id": "test_patient_002",
        "consent_verified": True,
        "survey_data": {
            "phq9_score": 20,
            "gad7_score": 18
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/screen",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Authenticated screening successful!")
            print(f"   Risk Score: {data['risk_score']['score']:.1f}")
            print(f"   Risk Level: {data['risk_score']['risk_level']}")
            return True
        else:
            print(f"❌ Authenticated screening failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Authenticated screening failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Mental Health Risk Assessment System - API Test")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ API is not running. Start it with: python run_api.py")
        sys.exit(1)
    
    # Test 2: Screening without auth
    test_screening_no_auth()
    
    # Test 3: Token generation
    token = test_token_generation()
    
    # Test 4: Screening with auth
    if token:
        test_screening_with_auth(token)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open frontend: python run_frontend.py")
    print("2. Visit: http://localhost:3000")
    print("3. Try the interactive interface!")

if __name__ == "__main__":
    main()
