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
        print(f"ADK Response structure: {response2_data}")
        
        try:
            # The response is an array of messages, we want the last one with 'text' content
            if isinstance(response2_data, list) and len(response2_data) > 0:
                # Look for the last message that has text content
                for message in reversed(response2_data):
                    if 'content' in message and 'parts' in message['content']:
                        for part in message['content']['parts']:
                            if 'text' in part:
                                text = part['text']
                                print(f"ADK Text: {text}")
                                return text
                
                # If no text found, return a fallback message
                print("No text content found in ADK response")
                return "Sorry, I couldn't process your request. Please try again."
            else:
                print(f"Unexpected response format: {response2_data}")
                return str(response2_data)
                
        except Exception as e:
            print(f"Error parsing ADK response: {e}")
            return f"Error parsing response: {str(e)}"
                
    else:
        print(f"Request to /run failed with status code {response2.status_code}")
        print(f"Response content: {response2.text}")
        return None
        