#STEP1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
#from insightface.data import get_image as ins_get_image



#STEP2 : 추론기 만들기
face = FaceAnalysis(providers=['CPUExecutionProvider'])
#CPU에서 model을 알아서 다 설치해준다
face.prepare(ctx_id=0, det_size=(640, 640))

from typing import Union
from fastapi import FastAPI, File, UploadFile
app = FastAPI()


import io
from PIL import Image
import numpy as np
@app.post("/files/")
async def create_file(file: bytes = File(), 
                      file2: bytes = File()): #파일 2개 받기

    #STEP3 #open cv에서 쓰는 방법 : 구조는 똑같고 이미지가 PIL, open cv를 요구하느냐에 따라 달라짐
    #img = cv2.imread("face.jpg", cv2.IMREAD_COLOR) 이 단계가 밑의 두 가지로 쪼개지는 것.
    nparr = np.fromstring(file, np.uint8)  # = pil_img = Image.open(io.BytesIO(file))
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    nparr2 = np.fromstring(file2, np.uint8) 
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    #STEP4 : 추론
    result = face.get(img)
    result2 = face.get(img2)
    print(result2)

    if len(result) == 0  or len(result2) == 0:
        return {"result": "fail"}
    
    face1 = result[0]
    face2 = result2[0]

    #얼굴이 닮았는지 아닌지 판별한다
    emb1 = face1.normed_embedding 
    emb2 = face2.normed_embedding

    sim = np.dot(emb1, emb2) #embedding객체를 가지고 dot( )으로 두 객체가 같은지 판별
		#Euclidean : 두 점 간의 거리를 재서 두 개체 간의 유사성을 판별한다
    #코싸인

    print(result[0].bbox)

    #STEP5
    return {"result":float(sim)}
    #numpy의 int객체때문에 float로 감싼다