from fastapi import FastAPI,File, UploadFile ,HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from inference import predict_image,convert_image_to_vector
from base import Patient,User,Exam,Base
from model import PatientCreate,UserCreate,ExamCreate,Credentials
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
# Connect to the database
engine = create_engine("sqlite:///brightM_database.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
    return {"message": path}

@app.post("/user")
async def create_user(user: UserCreate):
    session = SessionLocal()
    new_user = User(nom=user.nom, prenom=user.prenom, grade=user.grade, laboratoire=user.laboratoire, login=user.login,password=user.password)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    session.close()
    return {"id":new_user.idU,"nom":new_user.nom,"prenom":new_user.prenom,"grade":new_user.grade,"laboratoire":new_user.laboratoire,"login":new_user.login,"password":new_user.password}
@app.get("/user/{id}")
async def read_user(id: int):
    session = SessionLocal()
    user = session.query(User).filter(User.idU == id).first()
    session.close()
    return user.__dict__

@app.put("/user/{id}")
async def update_user(id: int, user: UserCreate):
    session = SessionLocal()
    user_to_update = session.query(User).filter(User.idU == id).first()
    user_to_update.nom = user.nom
    user_to_update.prenom = user.prenom
    user_to_update.grade = user.grade
    user_to_update.laboratoire = user.laboratoire
    user_to_update.login = user.login
    user_to_update.password = user.password
    session.commit()
    session.close()
    return {"message": "User updated successfully"}

@app.delete("/user/{id}")
async def delete_user(id: int):
    session = SessionLocal()
    user_to_delete = session.query(User).filter(User.idU == id).first()
    session.delete(user_to_delete)
    session.commit()
    session.close()
    return {"message": "User deleted successfully"}    
     
@app.post("/patient")
async def create_patient(patient: PatientCreate):
    session = SessionLocal()
    new_patient = Patient(nom=patient.nom, prenom=patient.prenom, sexe=patient.sexe,age=patient.age)
    session.add(new_patient)
    session.commit()
    session.refresh(new_patient)
    session.close()
    return {"idP":new_patient.idP,"nom":new_patient.nom,"prenom":new_patient.prenom,"sexe":new_patient.sexe,"age":new_patient.age}

@app.get("/patient/{id}")
async def read_patient(id: int):
    session = SessionLocal()
    patient = session.query(Patient).filter(Patient.idP == id).first()
    session.close()
    return patient.__dict__

@app.put("/patient/{id}")
async def update_patient(id: int, patient: PatientCreate):
    session = SessionLocal()
    patient_to_update = session.query(Patient).filter(Patient.idP == id).first()
    patient_to_update.nom = patient.nom
    patient_to_update.prenom = patient.prenom
    patient_to_update.sexe = patient.sexe
    session.commit()
    session.close()
    return {"message": "Patient updated successfully"}

@app.delete("/patient/{id}")
async def delete_patient(id: int):
    session = SessionLocal()
    patient_to_delete = session.query(Patient).filter(Patient.idP == id).first()
    session.delete(patient_to_delete)     
    session.commit()
    session.close()
    return {"message": "Patient deleted successfully"}
@app.post("/exam")
async def create_exam(exam: ExamCreate):
    session = SessionLocal()
    new_exam = Exam(date=exam.date, idP=exam.idP, idU=exam.idU,images=exam.images, result=exam.result)
    session.add(new_exam)
    session.commit()
    session.refresh(new_exam)
    session.close()
    return {"id":new_exam.id,"idP":new_exam.idP,"idU":new_exam.idU,"date":new_exam.date,"image":new_exam.images,"result":new_exam.result}

@app.get("/exam")
async def read_exam() -> List[dict]:
    session = SessionLocal()
    exams = session.query(Exam).all()
    session.close()
    return [ex.__dict__ for ex in exams]

@app.put("/exam/{id}")
async def update_exam(id: int, exam: ExamCreate):
    session = SessionLocal()
    exam_to_update = session.query(Exam).filter(Exam.id == id).first()
    exam_to_update.date = exam.date
    exam_to_update.idP = exam.idP
    exam_to_update.idU = exam.idU
    exam_to_update.symptome = exam.symptome
    exam_to_update.result = exam.result
    session.commit()
    session.close()
    return {"message": "Exam updated successfully"}

@app.delete("/exam/{id}")
async def delete_exam(id: int):
    session = SessionLocal()
    exam_to_delete = session.query(Exam).filter(Exam.id == id).first()
    session.delete(exam_to_delete)
    session.commit()
    session.close()
    return {"message": "Exam deleted successfully"}    
@app.post("/login")
async def login(credentials: Credentials):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.login == credentials.login).first()
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect login or password")
        if user.password != credentials.password:
            raise HTTPException(status_code=400, detail="Incorrect login or password")
        return user
    finally:
        session.close()