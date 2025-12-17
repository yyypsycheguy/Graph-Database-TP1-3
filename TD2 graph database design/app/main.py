from fastapi import FastAPI
from etl import etl

app = FastAPI()

@app.get("/etl")
def run_etl():
    return etl()

@app.get("/health")
def health_check():
    return {"status": "ok"}
