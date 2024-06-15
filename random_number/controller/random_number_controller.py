from fastapi import APIRouter, Depends

from random_number.service.random_number_service_impl import RandomNumberServiceImpl

randomNumberRouter = APIRouter()


async def getRandomNumberService() -> RandomNumberServiceImpl:
    return RandomNumberServiceImpl()


@randomNumberRouter.get("/draw")
async def drawRandomNumber(randomNumberService: RandomNumberServiceImpl = Depends(getRandomNumberService)):
    randomNumber = await randomNumberService.drawRandomNumber()
    return randomNumber
