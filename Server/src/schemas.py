from pydantic import BaseModel
from typing import List
import json

# Action = {
#   "action": str, 
#   "frames": [
#       "landmarks": [
#           {x: int, y: int, z: int}
#       ]
#   ]
# }

class Landmark(BaseModel):
    x: int = None
    y: int = None
    z: int = None
    class Config:
        orm_mode = True

class Frame(BaseModel):
    landmarks: List[Landmark]
    class Config:
        orm_mode = True

class Action(BaseModel):
    action: str = None
    frames: List[Frame]
    class Config:
        orm_mode = True

if __name__ == "__main__":
    jsonObj = Action.model_json_schema()
    
    #action = Action(id=1, action="hello", frames="world")
    #jsonObj = action.model_dump()

    string = json.dumps(jsonObj, indent=2)
    print(string)