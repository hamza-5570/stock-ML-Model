import os
from fastapi import HTTPException, Header
from dotenv import load_dotenv

# Load the token
load_dotenv()
SECRET_TOKEN = os.getenv("SECRET_TOKEN")

# Token Authentication
def verify_token(authorization: str = Header(None)):  
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token missing")
    if authorization != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True