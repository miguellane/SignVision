from database import Base
from typing import List
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Action = {
#   "action": str, 
#   "frames": [
#       "landmarks": [
#           {x: int, y: int, z: int}
#       ]
#   ]
# }

class Action(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(nullable=True, index=True)
    frames: Mapped[List["Frame"]] = relationship()

class Frame(Base):
    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey('actions.id'))
    landmarks: Mapped[List["Landmark"]] = relationship()

class Landmark(Base):
    __tablename__ = "landmarks"
    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[int] = mapped_column(nullable=True)
    y: Mapped[int] = mapped_column(nullable=True)
    z: Mapped[int] = mapped_column(nullable=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey('frames.id'))
    
