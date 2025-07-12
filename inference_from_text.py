"""
Emotion Classification using Transformers Pipeline
Model: Marc0222/emotion-distilbert-works

This script uses the transformers library directly to load and use the Marc0222 emotion model locally.
"""

from transformers import pipeline
from typing import List, Dict, Optional
import json

class EmotionClassifier:
    def __init__(self):
        """
        Initialize the emotion classifier with Marc0222 model using transformers pipeline
        """
        self.model_name = "Marc0222/emotion-distilbert-works"
        self.pipe = None
        
        # Emotion label mapping for Marc0222 model
        # Based on common emotion classification patterns
        self.label_mapping = {
            "LABEL_0": "sadness",
            "LABEL_1": "happiness", 
            "LABEL_2": "anger",
            "LABEL_3": "fear",
            "LABEL_4": "surprise",
            "LABEL_5": "neutral"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the emotion model"""
        try:
            print(f"🔄 Loading emotion model...")
            print(f"📦 Model: {self.model_name}")
            
            # Load the pipeline
            self.pipe = pipeline(
                "text-classification", 
                model=self.model_name,
                device=-1  # Use CPU, change to 0 for GPU if available
            )
            
            print("✅ Emotion model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading Emotion model: {str(e)}")
            self.pipe = None
    
    def classify_emotion(self, text: str) -> Dict:
        """
        Classify emotion in text using the Emotion model
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing emotion predictions with human-readable labels
        """
        if not self.pipe:
            return {"error": "Model not loaded"}
        
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            # Get predictions from the model
            results = self.pipe(text.strip())
            
            # Convert generic labels to emotion names
            if isinstance(results, list):
                mapped_results = []
                for result in results:
                    mapped_result = {
                        "label": self.label_mapping.get(result.get('label', ''), result.get('label', 'unknown')),
                        "score": result.get('score', 0)
                    }
                    mapped_results.append(mapped_result)
                return mapped_results
            else:
                # Single result
                mapped_result = {
                    "label": self.label_mapping.get(results.get('label', ''), results.get('label', 'unknown')),
                    "score": results.get('score', 0)
                }
                return [mapped_result]
                
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def get_top_emotion(self, text: str) -> Optional[str]:
        """
        Get the top emotion prediction for a text
        
        Args:
            text: Input text
            
        Returns:
            The emotion with highest confidence, or None if error
        """
        result = self.classify_emotion(text)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        if isinstance(result, list) and len(result) > 0:
            # Get the top emotion
            top_emotion = max(result, key=lambda x: x.get('score', 0))
            return top_emotion.get('label')
        
        return None
    
    def analyze_interview_emotions(self, messages: List[str]) -> Dict:
        """
        Analyze emotions across multiple messages and calculate overall percentages
        
        Args:
            messages: List of text messages from the interview
            
        Returns:
            Dictionary with aggregated emotion analysis
        """
        if not messages:
            return {"error": "No messages provided"}
        
        if not self.pipe:
            return {"error": "Model not loaded"}
        
        print(f"📊 Analyzing {len(messages)} messages for overall emotion patterns...")
        
        # Store all emotion results
        all_emotions = {}
        message_results = []
        successful_analyses = 0
        
        # Analyze each message
        for i, message in enumerate(messages):
            if not message or not message.strip():
                continue
                
            print(f"Processing message {i+1}/{len(messages)}...", end=" ")
            
            result = self.classify_emotion(message.strip())
            
            if "error" not in result and isinstance(result, list) and len(result) > 0:
                emotions = result
                message_results.append({
                    "message": message[:50] + "..." if len(message) > 50 else message,
                    "emotions": emotions
                })
                
                # Accumulate emotion scores
                for emotion in emotions:
                    label = emotion.get('label', 'unknown')
                    score = emotion.get('score', 0)
                    
                    if label not in all_emotions:
                        all_emotions[label] = []
                    all_emotions[label].append(score)
                
                successful_analyses += 1
                print("✅")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        if successful_analyses == 0:
            return {"error": "No messages could be analyzed successfully"}
        
        # Calculate aggregated percentages
        emotion_percentages = {}
        emotion_stats = {}
        
        for emotion, scores in all_emotions.items():
            # Calculate various statistics
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            # Calculate percentage (average score * 100)
            percentage = avg_score * 100
            
            emotion_percentages[emotion] = round(percentage, 2)
            emotion_stats[emotion] = {
                "average": round(avg_score, 4),
                "max": round(max_score, 4),
                "min": round(min_score, 4),
                "occurrences": len(scores),
                "percentage": round(percentage, 2)
            }
        
        # Sort by percentage (highest first)
        sorted_emotions = dict(sorted(emotion_percentages.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        # Calculate dominant emotion
        if emotion_percentages:
            dominant_emotion = max(emotion_percentages.items(), key=lambda x: x[1])
        else:
            dominant_emotion = ("unknown", 0)
        
        # Create summary
        summary = {
            "total_messages_analyzed": successful_analyses,
            "total_messages_provided": len(messages),
            "dominant_emotion": {
                "emotion": dominant_emotion[0],
                "percentage": dominant_emotion[1]
            },
            "emotion_percentages": sorted_emotions,
            "detailed_stats": emotion_stats,
            "message_breakdown": message_results
        }
        
        return summary
    
    def get_interview_emotion_summary(self, messages: List[str], top_n: int = 5) -> Dict:
        """
        Get a simplified summary of interview emotions
        
        Args:
            messages: List of text messages
            top_n: Number of top emotions to return
            
        Returns:
            Simplified emotion summary
        """
        analysis = self.analyze_interview_emotions(messages)
        
        if "error" in analysis:
            return analysis
        
        # Get top N emotions
        top_emotions = dict(list(analysis["emotion_percentages"].items())[:top_n])
        
        return {
            "total_messages": analysis["total_messages_analyzed"],
            "dominant_emotion": analysis["dominant_emotion"]["emotion"],
            "dominant_percentage": analysis["dominant_emotion"]["percentage"],
            "top_emotions": top_emotions,
            "emotion_breakdown": {
                emotion: f"{percentage}%" 
                for emotion, percentage in top_emotions.items()
            }
        }
    
    def get_emotion_breakdown_for_database(self, messages: List[str]) -> Dict:
        """
        Get emotion breakdown in the format expected by MongoDB
        
        Args:
            messages: List of text messages
            
        Returns:
            Dictionary with emotion percentages for all 6 emotions
        """
        analysis = self.analyze_interview_emotions(messages)
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        # Initialize all emotions to 0
        emotion_breakdown = {
            "sadness": 0.0,
            "happiness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 0.0
        }
        
        # Fill in the actual percentages from analysis
        for emotion, percentage in analysis["emotion_percentages"].items():
            if emotion in emotion_breakdown:
                emotion_breakdown[emotion] = round(percentage, 2)
        
        return emotion_breakdown

def interview_analysis(interview_messages: List[str]):
    """Interview emotion analysis"""
    print("\n🎯 Interview Analysis")
    print("-" * 35)
    
    classifier = EmotionClassifier()
    
    # Get simplified summary
    summary = classifier.get_interview_emotion_summary(interview_messages, top_n=5)
    
    if "error" not in summary:
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Interview Emotion Profile:")
        print(f"   Total Messages: {summary['total_messages']}")
        print(f"   Overall Tone: {summary['dominant_emotion']} ({summary['dominant_percentage']}%)")
        
        print(f"\n🎯 Top Emotions:")
        for emotion, percentage in summary['top_emotions'].items():
            print(f"   • {emotion.capitalize()}: {percentage}%")
        
        # Format for database storage
        interview_results = {
            "interview_id": "sample_interview_123",
            "candidate_emotion_analysis": {
                "dominant_emotion": summary['dominant_emotion'],
                "dominant_percentage": summary['dominant_percentage'],
                "emotion_breakdown": summary['top_emotions'],
                "total_messages_analyzed": summary['total_messages']
            }
        }
        
        print(f"\n💾 Data for Database Storage:")
        print(json.dumps(interview_results, indent=2))
        
    else:
        print(f"❌ Error: {summary['error']}")
