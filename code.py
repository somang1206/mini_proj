from fastapi import FastAPI, WebSocket, Request
import cv2
import base64
import numpy as np
import face_recognition
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize variables for face recognition

known_face_encodings = []
known_face_names = []

# Load known faces
somang_image = face_recognition.load_image_file("somang.jpg")
somang_face_encoding = face_recognition.face_encodings(somang_image)[0]
known_face_encodings.append(somang_face_encoding)
known_face_names.append("Somang Lee")

taehyun_image = face_recognition.load_image_file("taehyun.jpg")
taehyun_face_encoding = face_recognition.face_encodings(taehyun_image)[0]
known_face_encodings.append(taehyun_face_encoding)
known_face_names.append("Tae hyun Kim")

async def process_frames(websocket: WebSocket):
    await websocket.accept()
    video_capture = cv2.VideoCapture(0)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convert the frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer)
        print("buffer::::::::::::::" , buffer)

        # Send the frame to the client
        await websocket.send_bytes(frame_bytes)

@app.get("/client")
async def client(request: Request):
    return templates.TemplateResponse("video.html", {"request":request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print(f"client connected : {websocket.client}")
    await process_frames(websocket)
