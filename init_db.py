import sys
import os
from src.core.database import engine, Base
from src.api.auth.models import User, Token

# Create tables only if they do not exist
Base.metadata.create_all(bind=engine, checkfirst=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.database import engine, Base
from src.api.auth.models import User
from src.repository.prediction_repository import PredictionHistory

def init_db():
    """Create all database tables"""
    print("Creating database tables...")
    
    # Ensure all models are imported and referenced to include their tables
    _ = [User, Token, PredictionHistory]
    for model in _:
        model.__table__.create(bind=engine, checkfirst=True)

    print(f"Database URL: {engine.url}")

    print("Database tables created successfully!")
    print("Tables created:")
    for table in Base.metadata.tables:
        print(f"  - {table}")

    print("Registered tables in metadata:", Base.metadata.tables.keys())

if __name__ == "__main__":
    init_db()