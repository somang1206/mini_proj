from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import face_recognition
from io import BytesIO

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = FastAPI()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/")
async def upload_image(file: UploadFile = File(...)):
    # Check if a valid image file was uploaded
    if file.content_type.startswith('image/') and allowed_file(file.filename):
        # Read image file
        contents = await file.read()
        img_stream = BytesIO(contents)


        biden_image = face_recognition.load_image_file("winter.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        # Load the uploaded image file
        img = face_recognition.load_image_file(img_stream)
        # Get face encodings for any faces in the uploaded image
        unknown_face_encodings = face_recognition.face_encodings(img)

        face_found = False
        is_obama = False

        if len(unknown_face_encodings) > 0:
            face_found = True
            # See if the first face in the uploaded image matches the known face of Obama
            #match_results = face_recognition.compare_faces([biden_face_encoding], unknown_face_encodings[0])
            face_distances = face_recognition.face_distance([biden_face_encoding], unknown_face_encodings[0])
            if face_distances[0] < 0.3:
                is_obama = True

        # Return the result as json
        result = {
            "face_found_in_image": face_found,
            "is_picture_of_obama": is_obama
        }
        return JSONResponse(content=result)
    else:
        return JSONResponse(content={"error": "Invalid file. Please upload a valid image file."})




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)