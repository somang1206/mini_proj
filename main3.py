# from fastapi import FastAPI, WebSocket, Request
# import cv2
# import base64
# import numpy as np
# import face_recognition
# from fastapi.templating import Jinja2Templates


# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # Initialize variables for face recognition

# known_face_encodings = []
# known_face_names = []

# # Load known faces
# somang_image = face_recognition.load_image_file("somang.jpg")
# somang_face_encoding = face_recognition.face_encodings(somang_image)[0]
# known_face_encodings.append(somang_face_encoding)
# known_face_names.append("Somang Lee")

# taehyun_image = face_recognition.load_image_file("taehyun.jpg")
# taehyun_face_encoding = face_recognition.face_encodings(taehyun_image)[0]
# known_face_encodings.append(taehyun_face_encoding)
# known_face_names.append("Tae hyun Kim")

# async def process_frames(websocket: WebSocket):
#     await websocket.accept()
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         # Grab a single frame of video
#         ret, frame = video_capture.read()

#         # Resize frame of video to 1/4 size for faster face recognition processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # See if the face is a match for the known face(s)
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]

#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#             # Draw a label with a name below the face
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#         # Convert the frame to bytes
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = base64.b64encode(buffer)
#         print("buffer::::::::::::::" , buffer)

#         # Send the frame to the client
#         await websocket.send_bytes(frame_bytes)

# @app.get("/client")
# async def client(request: Request):
#     return templates.TemplateResponse("video.html", {"request":request})


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print(f"client connected : {websocket.client}")
#     await process_frames(websocket)

import shutil
from click import File
import face_recognition
from fastapi import FastAPI, Request, status, UploadFile, WebSocket, WebSocketDisconnect, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from sqlalchemy.orm import sessionmaker, Session
from typing import List
from database import SessionLocal, engine
import os
from sqlalchemy.orm import Session
from pydantic import BaseModel
import models
from models import Image 
from fastapi.staticfiles import StaticFiles




app = FastAPI()

#절대패스
abs_path = os.path.dirname(os.path.realpath(__file__))

save_dir = os.path.join(abs_path, "static")  # 이미지를 저장할 디렉토리의 절대 경로

templates = Jinja2Templates(directory=f"{abs_path}/templates")
app.mount("/static", StaticFiles(directory=f"{abs_path}/static"))


models.Base.metadata.create_all(bind=engine)

known_face_encodings = []
known_face_names = []






@app.post("/save-image")
async def add(fileName: str = Form(...), image: UploadFile = File(...)):
    save_path = os.path.join(save_dir, f"{fileName}.jpg")  # 이미지의 저장 경로 설정
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없는 경우 생성

    db = SessionLocal()

    file_contents = await image.read()

    with open(save_path, "wb") as f:
        f.write(file_contents)

    image_info = models.Image(name=fileName, path=file_contents)
    db.add(image_info)
    db.commit()

    return {"filename": fileName}
    # new_todo = models.Todo(task=title)

    #db테이블에 create
    # db.add(new_todo)

    # db.commit()
    # url = app.url_path_for("home")
    # return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()



def load_known_faces(db: Session):
    # taehyun_image = face_recognition.load_image_file("taehyun.jpg")
    # taehyun_face_encoding = face_recognition.face_encodings(taehyun_image)[0]

    # somang_image = face_recognition.load_image_file("somang.jpg")
    # somang_face_encoding = face_recognition.face_encodings(somang_image)[0]

    # dongjae_image = face_recognition.load_image_file("dongjae.jpg")
    # dongjea_face_encoding = face_recognition.face_encodings(dongjae_image)[0]

    # sanghoon_image = face_recognition.load_image_file("sanghoon.jpg")
    # sanghoon_face_encoding = face_recognition.face_encodings(sanghoon_image)[0]

    # miae_image = face_recognition.load_image_file("miae.jpg")
    # miae_face_encoding = face_recognition.face_encodings(miae_image)[0]

    # known_face_encodings.extend([taehyun_face_encoding, somang_face_encoding])
    # known_face_names.extend(["taehyun", "somang"])
    db = SessionLocal()
    images = db.query(Image).all()

    known_face_encodings = []
    known_face_names = []

    # 각 이미지에 대해 얼굴 인코딩을 추출합니다.
    for image in images:
        # 이미지 데이터를 numpy 배열로 변환합니다.
        image_path = image.path
        with open(image_path, "rb") as f:
            image_data = f.read()
        nparr = np.frombuffer(image_data, np.uint8)
        print("image.path!!!!!!!!!" ,image.path)
        
        # OpenCV를 사용하여 이미지를 로드합니다.
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 얼굴 인코딩을 추출합니다.
        face_encoding = face_recognition.face_encodings(img_np)[0]
        
        # 추출된 얼굴 인코딩과 이미지의 이름을 리스트에 추가합니다.
        known_face_encodings.append(face_encoding)
        known_face_names.append(image.name)

    return known_face_encodings, known_face_names

@app.on_event("startup")
def startup_event():
       # Load known faces at startup
    db = SessionLocal()
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces(db)
    db.close()

@app.get("/detect")
async def read_root(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})

async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Initialize variables for comparison
                name = "denied"
                min_distance = 0.4  # Set a threshold for face distance

                # Compare face encoding with known faces
                for known_encoding, known_name in zip(known_face_encodings, known_face_names):
                    face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    if face_distance < min_distance:
                        min_distance = face_distance
                        name = "access"

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name == "access":
                    cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 1.0, (255, 255, 255), 1)

            _, jpeg = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(jpeg.tobytes())

    except WebSocketDisconnect:
        pass
    finally:
        video_capture.release()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)
    

@app.get("/capture", response_class=HTMLResponse)
async def read_capture(request: Request):
    return templates.TemplateResponse("capture.html", {"request": request})