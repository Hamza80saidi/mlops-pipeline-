from typing import Optional, List
from sqlalchemy.orm import Session
from .base_repository import BaseRepository
from ..api.auth.models import User

class UserRepository(BaseRepository[User]):
    """Repository for user operations"""
    
    def __init__(self, db_session: Session):
        super().__init__(db_session)
    
    def create(self, user: User) -> User:
        self.db_session.add(user)
        self.db_session.commit()
        self.db_session.refresh(user)
        return user
    
    def get(self, id: int) -> Optional[User]:
        return self.db_session.query(User).filter(User.id == id).first()
    
    def get_by_username(self, username: str) -> Optional[User]:
        return self.db_session.query(User).filter(User.username == username).first()
    
    def update(self, user: User) -> User:
        self.db_session.commit()
        self.db_session.refresh(user)
        return user
    
    def delete(self, id: int) -> bool:
        user = self.get(id)
        if user:
            self.db_session.delete(user)
            self.db_session.commit()
            return True
        return False
    
    def list(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db_session.query(User).offset(skip).limit(limit).all()