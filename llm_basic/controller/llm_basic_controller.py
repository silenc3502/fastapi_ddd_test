import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

import tensorflow as tf
import numpy as np

import openai

llmBasicRouter = APIRouter()


api_key = ''
openai_api_url = 'https://api.openai.com/v1/completions'

# 입력 데이터를 위한 Pydantic 모델 정의
class TextPrompt(BaseModel):
    prompt: str

# 텍스트 생성 요청을 보내는 함수
async def generate_text(prompt: str, max_tokens: int = 50) -> str:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'prompt': prompt,
        'max_tokens': max_tokens
    }
    async with httpx.AsyncClient() as client:
        print("start httpx.AsyncClient()")
        try:
            response = await client.post(openai_api_url, headers=headers, json=data)
            print("after post()")
            response.raise_for_status()
            print("goto get response")
            return response.json()['choices'][0]['text'].strip()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e}")
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {e}")
        except (httpx.RequestError, ValueError) as e:
            print(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")


# POST 요청을 처리하는 라우터
@llmBasicRouter.post('/lets-talk')
async def lets_talk(prompt_data: TextPrompt):
    try:
        generated_text = await generate_text(prompt_data.prompt)
        return {'generated_text': generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
