from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms, models
from pydantic import BaseModel
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production to allow only specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Model and Preprocessing Setup ----------------------

def load_emotion_model(model_path='trained_models/resnet18_with_stepdecay_20.pth'):
    """
    Load a pre-trained ResNet18 model for emotion detection.
    Assumes the model was trained with 7 emotion classes.
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

emotion_model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transformation pipeline for the PyTorch model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet standard input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------- MediaPipe Setup ----------------------

# Face Mesh (for facial landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face Detection (for cropping the face for emotion detection)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ---------------------- Landmark Indices ----------------------

# Eye landmarks indices
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_EYE_LEFT = 263
LEFT_EYE_RIGHT = 362

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_EYE_LEFT = 133
RIGHT_EYE_RIGHT = 33

# Landmarks used for face orientation
NOSE_TIP = 4
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# ---------------------- Helper Functions ----------------------

def calculate_ear(landmarks, eye_pts):
    """
    Calculate the Eye Aspect Ratio (EAR) from the given landmarks.
    eye_pts: list of four landmark indices [top, bottom, left, right]
    """
    top = landmarks[eye_pts[0]]
    bottom = landmarks[eye_pts[1]]
    left = landmarks[eye_pts[2]]
    right = landmarks[eye_pts[3]]
    
    vertical_dist = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    horizontal_dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
    
    ear = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    return ear

def detect_face_orientation(landmarks):
    """
    Determine face orientation using key facial landmarks.
    Returns:
      - primary_dir: the main direction (Left, Right, Up, Down, or Center)
      - is_looking_forward: True if looking forward
      - is_looking_down: True if looking downward
    """
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    
    eye_midpoint_x = (left_eye.x + right_eye.x) / 2
    eye_midpoint_y = (left_eye.y + right_eye.y) / 2
    
    # Compute direction vector from eyes to nose
    dir_x = nose.x - eye_midpoint_x
    dir_y = nose.y - eye_midpoint_y
    dir_z = nose.z - ((left_eye.z + right_eye.z) / 2)
    
    # Determine horizontal direction
    horizontal_threshold = 0.03
    if dir_x > horizontal_threshold:
        horizontal_dir = "Right"
    elif dir_x < -horizontal_threshold:
        horizontal_dir = "Left"
    else:
        horizontal_dir = "Center"
    
    # Determine vertical direction
    vertical_threshold = 0.02
    if dir_y > vertical_threshold:
        vertical_dir = "Down"
    elif dir_y < -vertical_threshold:
        vertical_dir = "Up"
    else:
        vertical_dir = "Center"
    
    # Determine primary direction
    if horizontal_dir != "Center":
        primary_dir = horizontal_dir
    elif vertical_dir != "Center":
        primary_dir = vertical_dir
    else:
        primary_dir = "Center"
    
    is_looking_forward = (horizontal_dir == "Center" and vertical_dir != "Down")
    is_looking_down = (vertical_dir == "Down")
    
    return primary_dir, is_looking_forward, is_looking_down

def detect_emotion(face_img):
    """
    Detect emotion using the pre-trained PyTorch model.
    Returns the emotion label and the confidence score.
    """
    if face_img.shape[0] <= 0 or face_img.shape[1] <= 0:
        return "Unknown", 0.0
    
    input_tensor = transform(face_img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    emotion_model.to(device)
    
    with torch.no_grad():
        outputs = emotion_model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    confidence, predicted = torch.max(probabilities, 1)
    emotion_idx = predicted.item()
    emotion = emotion_labels[emotion_idx]
    confidence_value = confidence.item()
    
    return emotion, confidence_value

def classify_engagement(emotion, eye_state, looking_direction):
    """
    Classify engagement state based on emotion, eye state, and looking direction.
    """
    positive_emotions = ['Happy', 'Neutral', 'Surprise']
    negative_emotions = ['Sad', 'Disgust', 'Fear', 'Angry']
    
    if emotion in positive_emotions and eye_state == "Open" and looking_direction == "Forward":
        return "Focused"
    elif emotion in negative_emotions and (eye_state in ["Open", "Closed"]) and looking_direction == "Away":
        return "Not Focused"
    elif eye_state in ["Half-Closed", "Closed"] and looking_direction == "Down":
        return "Neutral"
    else:
        if looking_direction == "Forward" and eye_state == "Open":
            return "Focused"
        elif looking_direction == "Down":
            return "Neutral"
        else:
            return "Not Focused"

# ---------------------- Response Model ----------------------

class AnalysisResult(BaseModel):
    emotion: str
    emotion_confidence: float
    eye_state: str
    looking_direction: str
    engagement: str

# ---------------------- FastAPI Endpoint ----------------------

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image file upload, processes the image to detect facial landmarks,
    calculates the eye state and face orientation, runs emotion detection, and classifies engagement.
    """
    try:
        # Read image file contents and decode to an OpenCV image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert image to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh for landmarks
        mesh_results = face_mesh.process(image_rgb)
        
        # Process with face detection for cropping the face for emotion detection
        detection_results = face_detection.process(image_rgb)
        
        # Initialize default values
        face_found = False
        eye_state = "Unknown"
        looking_direction = "Unknown"
        emotion = "Unknown"
        emotion_confidence = 0.0
        
        # If landmarks are detected, compute EAR and orientation
        if mesh_results.multi_face_landmarks:
            face_found = True
            face_landmarks = mesh_results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            left_ear = calculate_ear(landmarks, [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT])
            right_ear = calculate_ear(landmarks, [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT])
            ear = (left_ear + right_ear) / 2.0
            
            ear_threshold = 0.22
            half_closed_threshold = 0.28
            if ear < ear_threshold:
                eye_state = "Closed"
            elif ear < half_closed_threshold:
                eye_state = "Half-Closed"
            else:
                eye_state = "Open"
            
            primary_dir, is_looking_forward, is_looking_down = detect_face_orientation(landmarks)
            if is_looking_forward:
                looking_direction = "Forward"
            elif is_looking_down:
                looking_direction = "Down"
            else:
                looking_direction = "Away"
        
        # Use face detection to crop the face and run emotion detection
        if detection_results.detections:
            detection = detection_results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            x, y = max(0, x), max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            if w > 0 and h > 0:
                face_img = image[y:y+h, x:x+w]
                emotion, emotion_confidence = detect_emotion(face_img)
        
        # Classify the engagement based on the computed metrics
        engagement = classify_engagement(emotion, eye_state, looking_direction)
        
        return AnalysisResult(
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            eye_state=eye_state,
            looking_direction=looking_direction,
            engagement=engagement
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Main ----------------------

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000) 
