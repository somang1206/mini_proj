from sqlalchemy import Column, Integer, Boolean, String, LargeBinary
from database import Base

class Image(Base):
	__tablename__ = 'image'
	index = Column(Integer, primary_key=True)
	name = Column(String(100))
	path = Column(LargeBinary)