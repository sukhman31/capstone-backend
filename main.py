from fastapi import FastAPI
from long_format import output

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.get("/long-format")
async def long_format(document, question):
    value = await output(document,question)
    return {"message": value}