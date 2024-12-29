from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from api.model import Model

app = FastAPI(title="a simple text classification")

loaded_model = Model()

# pred = loaded_model.predict_single("I hate this product.")
# print(f"Got: {pred}")

class Text(BaseModel):
    model_config = {"extra": "forbid"}
    text: str = Query(None, min_length=10, max_length=40)

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

@app.post("/predict-single")
def predict_single(RequestBody: Text):
    try:
        predicted_intent = loaded_model.predict_single(RequestBody.text)
        return {"predicted_intent": predicted_intent}
    except Exception as e:
        print(e)
        return None