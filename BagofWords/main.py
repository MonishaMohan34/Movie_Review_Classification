
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
from pathlib import Path


try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {str(e)}")


app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, review: str = Form(...)):
    try:
        
        vectorized_input = vectorizer.transform([review])
        
        prediction = model.predict(vectorized_input)[0]
        return templates.TemplateResponse(
            "result.html", {"request": request, "review": review, "sentiment": prediction}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html", {"request": request, "review": review, "sentiment": f"Error: {str(e)}"}
        )
