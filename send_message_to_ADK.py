import requests

def send_message_to_adk(user, message):
    payload = {
        "app_name": "CODEEVAL",
        "user_id": user.user_id,
        "session_id": user.session_id,
        "new_message": {
            "role": "user",
            "parts": [{
                "text": message
            }]
        }
    }
    response2 = requests.post(url="http://localhost:8000/run", json=payload)
    if response2.status_code == 200:
        response2_data = response2.json()
        text = response2_data[0]['content']['parts'][0]['text']
        print(text)
        return text
    else:
        print(f"Request to /run failed with status code {response2.status_code}")
        print(f"Response content: {response2.text}")
        return None
        