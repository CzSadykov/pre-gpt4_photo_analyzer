from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import (BlipProcessor,
                          BlipForQuestionAnswering)
from PIL import Image
from io import BytesIO

import uvicorn
import requests
import numpy as np


app = FastAPI()
processor_qa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_qa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


class Question(BaseModel):
    id: str
    question: str


class Query(BaseModel):
    photo_url: str
    questions: List[Question]


class Answer(BaseModel):
    id: str
    answer: str


class Response(BaseModel):
    answers: List[Answer]


@app.post("/analyze_photo", response_model=Response)
async def analyze_photo(query: Query):
    try:
        response = requests.get(query.photo_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = np.array(image)

        answers_list = []

        for q in query.questions:
            input_data = processor_qa(image, q.question, return_tensors="pt")
            output = model_qa.generate(**input_data)
            answer_text = processor_qa.decode(
                output[0],
                skip_special_tokens=True
                )

            answers_list.append({"id": q.id, "answer": answer_text})

        return {"answers": answers_list}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
