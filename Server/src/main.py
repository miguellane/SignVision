# Framework and template
from fastapi import FastAPI, Depends, Request, Form, status
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")
# Database
import Server.src.schemas as schemas, Server.src.models as models
from database import engine, get_db
from sqlalchemy.orm import Session
models.Base.metadata.create_all(bind=engine)
# SignVision Model
import SVmodel

app = FastAPI()

@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    #Display all the actions in the database
    actions = db.query(models.Action).all()
    return templates.TemplateResponse("base.html", {"request": request, "actions": actions})

@app.post("/add")
async def add(payload: schemas.Action, db: Session = Depends(get_db)):
    if payload.action:
        print("Save")
        new_action = models.Action(action=payload.action)
        for frame in payload.frames:
            new_frame = models.Frame()
            for landmark in frame.landmarks:
                new_landmark = models.Landmark(x=landmark.x, y=landmark.y, z=landmark.z)
                new_frame.landmarks.append(new_landmark)
            new_action.frames.append(new_frame)
        db.add(new_action)
        db.commit()
    else:
        print("Classify")
        print(payload.action)

    return JSONResponse({"classification" : payload.action})
