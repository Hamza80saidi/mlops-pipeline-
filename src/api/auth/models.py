from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from src.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Token(Base):
    __tablename__ = "tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    access_token = Column(String, unique=True, index=True)
    refresh_token = Column(String, unique=True, index=True)
    token_type = Column(String)
    user_id = Column(Integer)
    expires_at = Column(DateTime)
