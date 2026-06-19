from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is running"}
@app.get("/hello")
def hello():
    return {"message": "Hello World"}
@app.get("/user/{name}")
def get_user(name: str):
    return {"user": name}
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

@app.post("/user")
def create_user(user: User):
    return {"message": "User created", "data": user}