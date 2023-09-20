from transformers import (BlipProcessor,
                          BlipForQuestionAnswering)
from PIL import Image
from typing import List
import numpy as np
from io import BytesIO

processor_qa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_qa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


class PhotoAnalyzer:

    @classmethod
    def analyze_photo(cls, message, bot, questions: List = None):

        if questions is None:
            questions = [
                'Are there any persons on the photo?',
                'How many persons are on the photo?',
                'What object is on the photo?',
                'Is it indoors or outdoors?',
                'Is it day or night?',
                'What is the mood of the photo?'
            ]

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image = Image.open(BytesIO(downloaded_file)).convert('RGB')
        image = np.array(image)

        photo_qa = {}

        for question in questions:
            input = processor_qa(image, question, return_tensors="pt")
            output = model_qa.generate(**input)
            answer = processor_qa.decode(output[0], skip_special_tokens=True)
            photo_qa[question] = answer
            if answer.strip():
                photo_qa[question] = answer

        return photo_qa

    def format_photo_info(self, photo_qa: dict) -> str:
        return ''.join(
            f'Question:{key}\nAnswer:{value}\n'
            for key, value in photo_qa.items()
            )
