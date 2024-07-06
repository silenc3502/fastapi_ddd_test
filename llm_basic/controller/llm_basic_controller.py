import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드
load_dotenv()

llmBasicRouter = APIRouter()

# 환경 변수에서 API 키를 가져옴
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

openai_api_url = 'https://api.openai.com/v1/chat/completions'

# 입력 데이터를 위한 Pydantic 모델 정의
class TextPrompt(BaseModel):
    prompt: str

# 텍스트 생성 요청을 보내는 함수
async def generate_text(prompt: str) -> str:
    print(f"prompt: {prompt}")
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-4',
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(openai_api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e}")
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {e}")
        except (httpx.RequestError, ValueError) as e:
            print(f"Request error: {e}")
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

# POST 요청을 처리하는 라우터
@llmBasicRouter.post('/lets-talk')
async def lets_talk(prompt_data: TextPrompt):
    try:
        generated_text = await generate_text(prompt_data.prompt)
        return {'generated_text': generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
