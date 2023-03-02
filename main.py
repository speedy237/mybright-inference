from fastapi import FastAPI,File, UploadFile ,HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from typing import List
from inference import predict_image,convert_image_to_vector
from predict_lung_seg import predict
import os
from pathlib import Path
import tempfile
from typing import List



app = FastAPI()
origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"welcome:" "bright-medical"}
@app.get("/test")
def test_api():
    return {"welcome bright-medical"}    
@app.post("/predict")
async def predicts(file: UploadFile):
    print('image conversion')
     # Convertir l'image téléchargée en vecteur
    image_vector = convert_image_to_vector(file.file)
    # Effectuer la prédiction
    prediction = predict_image(image_vector)
    print(prediction)
    response={"prediction": prediction.tolist()}
    return response
@app.post("/segmentation")
async def image_segmentation(file:UploadFile):
    filename = file.filename
    _, ext = os.path.splitext(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
        temp.write(await file.read())
        temp.seek(0)
        origin_filename = temp.name
    path="segmentation_"+filename 
    rep_courrant=os.getcwd()
    #rep_cyble= "c://Users/JORDAN/Bright-Medicals/front/mybright/src/assets"   
    #desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    chemin_relatif = os.path.join(rep_courrant,"results")
    path_save_image = os.path.join(chemin_relatif, path)    
    print("file path")
    print(origin_filename)
    model_name = Path("unet-6v.pt")
    predict(model_name, origin_filename, path_save_image)
    return {"message": path_save_image}

 