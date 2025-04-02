from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from engagement_tracker import EngagementTracker

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the engagement tracker
tracker = EngagementTracker()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Analyze frame
    analysis = tracker.analyze_frame(img)
    
    return {
        "emotion": analysis['emotion'],
        "confidence": float(analysis['emotion_confidence']),
        "eye_state": analysis['eye_state'],
        "looking_direction": analysis['looking_direction'],
        "engagement": analysis['engagement']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 