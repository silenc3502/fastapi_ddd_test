from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from word_cloud.service.word_cloud_service_impl import WordCloudServiceImpl

wordCloudRouter = APIRouter()


async def injectWordCloudService() -> WordCloudServiceImpl:
    return WordCloudServiceImpl()


@wordCloudRouter.get("/word-cloud",)
async def createWordCloud(wordCloudService: WordCloudServiceImpl =
                               Depends(injectWordCloudService)):

    img_base64 = wordCloudService.generateWordCloud()
    return JSONResponse(content={"wordcloud": img_base64})
