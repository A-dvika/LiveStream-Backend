import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
from datetime import datetime
from PIL import Image
import io

class EngagementTracker:
    def __init__(self, model_path='trained_models/resnet18_with_stepdecay_20.pth'):
        # Initialize emotion detection
        self.emotion_model = self.load_emotion_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = {
            'TOP': 386,
            'BOTTOM': 374,
            'LEFT': 263,
            'RIGHT': 362
        }
        
        self.RIGHT_EYE_INDICES = {
            'TOP': 159,
            'BOTTOM': 145,
            'LEFT': 133,
            'RIGHT': 33
        }
        
        # Thresholds
        self.EAR_THRESHOLD = 0.22
        self.HALF_CLOSED_THRESHOLD = 0.28
    
    def load_emotion_model(self, model_path):
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def calculate_ear(self, landmarks, eye_pts):
        """Calculate Eye Aspect Ratio"""
        top = landmarks[eye_pts['TOP']]
        bottom = landmarks[eye_pts['BOTTOM']]
        left = landmarks[eye_pts['LEFT']]
        right = landmarks[eye_pts['RIGHT']]
        
        vertical_dist = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        horizontal_dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
        
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    
    def detect_face_orientation(self, landmarks):
        """Improved face orientation detection"""
        try:
            nose = landmarks[4]  # NOSE_TIP
            left_eye = landmarks[33]  # LEFT_EYE
            right_eye = landmarks[263]  # RIGHT_EYE

            # Calculate the midpoint of the eyes
            eye_midpoint_x = (left_eye.x + right_eye.x) / 2
            eye_midpoint_y = (left_eye.y + right_eye.y) / 2
            eye_midpoint_z = (left_eye.z + right_eye.z) / 2

            # Calculate direction differences
            dir_x = nose.x - eye_midpoint_x
            dir_y = nose.y - eye_midpoint_y
            dir_z = nose.z - eye_midpoint_z

            # Adjusted thresholds
            if abs(dir_z) < 0.1:  # Reduced threshold for Z-axis (depth)
                if abs(dir_x) < 0.1:  # Reduced threshold for X-axis (horizontal)
                    if dir_y > 0.05:
                        return "Down"
                    elif dir_y < -0.05:
                        return "Up"
                    else:
                        return "Forward"
            
            # More specific direction classification based on X-axis deviation
            if dir_x > 0.1:
                return "Right"
            elif dir_x < -0.1:
                return "Left"

            return "Away"  # Default to "Away" if thresholds not met
        except Exception as e:
            print(f"Error in face orientation detection: {e}")
            return "Unknown"

    
    def detect_emotion(self, face_img):
        """Detect emotion using ResNet model"""
        try:
            if face_img.shape[0] <= 0 or face_img.shape[1] <= 0:
                return "Unknown", 0.0
            
            # Convert to PIL and apply transformations
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.emotion_model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion = self.emotion_labels[predicted.item()]
                confidence_value = confidence.item()
                
                return emotion, confidence_value
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0
    
    def classify_engagement(self, emotion, eye_state, looking_direction):
        """Classify engagement level"""
        positive_emotions = ['Happy', 'Neutral', 'Surprise']
        negative_emotions = ['Sad', 'Disgust', 'Fear', 'Angry']
        
        # Camera off case
        if not any([emotion, eye_state, looking_direction]) or all(x == "Unknown" for x in [emotion, eye_state, looking_direction]):
            return "Not Focused"
        
        # If we can detect any facial features, consider them partially engaged
        if emotion != "Unknown" or eye_state != "Unknown" or looking_direction != "Unknown":
            # Positive emotion cases
            if emotion in positive_emotions:
                if eye_state == "Open":
                    return "Focused"  # Always focused if positive and eyes open
                elif eye_state == "Half-Closed":
                    return "Neutral"
            
            # Neutral state cases
            if eye_state == "Open":
                if looking_direction != "Away":
                    return "Focused"  # If eyes are open and not looking away, they're focused
                else:
                    return "Neutral"  # Give benefit of doubt when looking away briefly
            
            # Default to Neutral if we can detect the face but conditions aren't ideal
            return "Neutral"
        
        return "Not Focused"
    
    def analyze_frame(self, frame):
        """Analyze a single frame and return engagement metrics"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.mp_face_mesh.process(rgb_frame)
        
        # Default values
        analysis = {
            'face_detected': False,
            'emotion': 'Unknown',
            'emotion_confidence': 0.0,
            'eye_state': 'Unknown',
            'looking_direction': 'Unknown',
            'engagement': 'Not Focused'
        }
        
        if results.multi_face_landmarks:
            analysis['face_detected'] = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0
            
            # Determine eye state
            if ear < self.EAR_THRESHOLD:
                analysis['eye_state'] = "Closed"
            elif ear < self.HALF_CLOSED_THRESHOLD:
                analysis['eye_state'] = "Half-Closed"
            else:
                analysis['eye_state'] = "Open"
            
            # Get looking direction
            analysis['looking_direction'] = self.detect_face_orientation(landmarks)
            
            # Get face region for emotion detection
            ih, iw, _ = frame.shape
            x_min = min(landmark.x for landmark in landmarks)
            x_max = max(landmark.x for landmark in landmarks)
            y_min = min(landmark.y for landmark in landmarks)
            y_max = max(landmark.y for landmark in landmarks)
            
            x1, y1 = int(x_min * iw), int(y_min * ih)
            x2, y2 = int(x_max * iw), int(y_max * ih)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            
            if x2 > x1 and y2 > y1:
                face_img = frame[y1:y2, x1:x2]
                emotion, confidence = self.detect_emotion(face_img)
                analysis['emotion'] = emotion
                analysis['emotion_confidence'] = confidence
            
            # Classify engagement
            analysis['engagement'] = self.classify_engagement(
                analysis['emotion'],
                analysis['eye_state'],
                analysis['looking_direction']
            )
        
        return analysis 