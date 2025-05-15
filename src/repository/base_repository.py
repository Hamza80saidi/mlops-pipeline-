from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from sqlalchemy.orm import Session

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Abstract base repository"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    @abstractmethod
    def create(self, obj: T) -> T:
        """Create a new object"""
        pass
    
    @abstractmethod
    def get(self, id: int) -> Optional[T]:
        """Get object by ID"""
        pass
    
    @abstractmethod
    def update(self, obj: T) -> T:
        """Update an object"""
        pass
    
    @abstractmethod
    def delete(self, id: int) -> bool:
        """Delete an object"""
        pass
    
    @abstractmethod
    def list(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List objects with pagination"""
        pass