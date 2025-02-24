from .session import get_session
from .models import Base

def init_db():
    # Initialize the database and create tables
    Base.metadata.create_all(bind=get_session().bind)