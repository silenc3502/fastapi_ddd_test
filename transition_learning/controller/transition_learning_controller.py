from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# APIRouter 인스턴스 생성
transitionLearningRouter = APIRouter()

# 사전 학습된 모델과 토크나이저 로드
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 모델을 평가 모드로 설정
model.eval()

# 감정 분석 모델 로드
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.eval()

# 문장 임베딩 모델 로드
embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')

# 텍스트 생성 모델 로드
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

# 요청 바디 모델 정의
class TextPair(BaseModel):
    sentence1: str
    sentence2: str

class Text(BaseModel):
    sentence: str

# 헬스 체크 엔드포인트
@transitionLearningRouter.get("/")
def read_root():
    return {"message": "BERT-based Text Classification API"}

# 텍스트 분류를 위한 엔드포인트
@transitionLearningRouter.post("/transition-learning-predict/")
def predict(text_pair: TextPair):
    # 입력 텍스트를 토크나이즈
    inputs = tokenizer(text_pair.sentence1, text_pair.sentence2, return_tensors="pt", truncation=True)

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()

    # 예측 결과 반환
    if predictions == 1:
        result = "Paraphrase"
    else:
        result = "Not a Paraphrase"

    return {"sentence1": text_pair.sentence1, "sentence2": text_pair.sentence2, "prediction": result}

# 감정 분석 엔드포인트
@transitionLearningRouter.post("/sentiment-analysis/")
def sentiment_analysis(text: Text):
    inputs = sentiment_tokenizer(text.sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()

    sentiments = ["very negative", "negative", "neutral", "positive", "very positive"]
    sentiment = sentiments[predictions]

    return {"sentence": text.sentence, "sentiment": sentiment}


# 문장 임베딩 엔드포인트
@transitionLearningRouter.post("/sentence-embedding/")
def sentence_embedding(text: Text):
    embeddings = embedding_model.encode(text.sentence)

    return {"sentence": text.sentence, "embedding": embeddings.tolist()}


# 텍스트 생성 엔드포인트
@transitionLearningRouter.post("/generate-text/")
def generate_text(text: Text):
    inputs = gpt2_tokenizer.encode(text.sentence, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"input_sentence": text.sentence, "generated_text": generated_text}