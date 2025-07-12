#!/usr/bin/env python3
"""
Test script for emotion extraction API endpoint
"""
import requests
import json

def test_emotion_extraction():
    """Test the emotion extraction endpoint"""
    
    # Test messages (simulating candidate responses)
    test_messages = [
        "I'm really excited about this opportunity and feel confident about my skills.",
        "I'm a bit nervous but I think I can handle the technical challenges.",
        "That question makes me worried I might not be qualified enough.",
        "I'm happy to discuss my experience with Python and machine learning.",
        "Sometimes I feel frustrated when debugging complex issues.",
        "I remain calm under pressure and enjoy problem-solving."
    ]
    
    # Prepare the request
    url = "http://localhost:5000/extract_emotion"
    payload = {
        "messages": test_messages
    }
    
    try:
        print("🧪 Testing emotion extraction endpoint...")
        print(f"📡 Sending request to: {url}")
        print(f"📝 Test messages ({len(test_messages)} messages):")
        for i, msg in enumerate(test_messages, 1):
            print(f"   {i}. {msg}")
        
        # Make the request
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Response:")
            print(json.dumps(result, indent=2))
            
            if "emotion_analysis" in result:
                emotions = result["emotion_analysis"]
                print(f"\n🎯 Emotion Breakdown:")
                for emotion, percentage in emotions.items():
                    print(f"   • {emotion.capitalize()}: {percentage}%")
        else:
            print("❌ Error!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_emotion_extraction()
