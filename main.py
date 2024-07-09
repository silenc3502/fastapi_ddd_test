import asyncio
import json
import os

import nltk
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError
from dotenv import load_dotenv
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from convolution_neural_network.controller.cnn_controller import convolutionNeuralNetworkRouter
from decision_forest.controller.decision_forest_controller import decisionForestRouter
from exponential_regression.controller.exponential_regression_controller import exponentialRegressionRouter
from gradient_descent.controller.gradient_descent_controller import gradientDescentRouter
from kmeans_analysis.controller.kmeans_controller import kmeansRouter
from llm_basic.controller.llm_basic_controller import llmBasicRouter
from logistic_regression.controller.logistic_regression_controller import logisticRegressionRouter
from natural_language_processing.controller.natural_language_processing_controller import \
    naturalLanguageProcessingRouter
from openai_fine_tuning.controller.oft_controller import openAiFineTuningTestRouter
from orders_analysis.controller.orders_analysis_controller import ordersAnalysisRouter
from pca.controller.pca_controller import pcaRouter
from polynomial_regression.controller.polynomial_regression_controller import polynomialRegressionRouter
from post.controller.post_controller import post_router
from async_db.database import getMysqlPool
from random_forest.controller.random_forest_controller import randomForestRouter
from random_number.controller.random_number_controller import randomNumberRouter
from train_test_evaluation.controller.train_test_evaluation_controller import trainTestEvaluationRouter
from word_cloud.controller.word_cloud_controller import wordCloudRouter
from async_db.database import createTableIfNecessary


async def create_kafka_topics():
    admin_client = AIOKafkaAdminClient(
        bootstrap_servers='localhost:9092',
        loop=asyncio.get_running_loop()
    )

    try:
        await admin_client.start()

        topics = [
            NewTopic(
                "test-topic",
                num_partitions=1,
                replication_factor=1,
            ),
            NewTopic(
                "completion_topic",
                num_partitions=1,
                replication_factor=1,
            )
        ]

        for topic in topics:
            try:
                await admin_client.create_topics([topic])
            except TopicAlreadyExistsError:
                print(f"Topic '{topic.name}' already exists, skipping creation")

    except Exception as e:
        print(f"Failed to create Kafka topics: {e}")
    finally:
        await admin_client.close()



async def lifespan(app: FastAPI):
    app.state.db_pool = await getMysqlPool()
    await createTableIfNecessary(app.state.db_pool)

    app.state.stop_event = asyncio.Event()

    app.state.kafka_producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
        client_id="fastapi-kafka-producer"
    )

    app.state.kafka_consumer = AIOKafkaConsumer(
        'completion_topic',
        bootstrap_servers='localhost:9092',
        group_id="my_group",
        client_id="fastapi-kafka-consumer"
    )

    app.state.kafka_test_consumer = AIOKafkaConsumer(
        "test-topic",
        bootstrap_servers='localhost:9092',
        group_id="another_group",
        client_id="fastapi-kafka-consumer"
    )

    await app.state.kafka_producer.start()
    await app.state.kafka_consumer.start()
    await app.state.kafka_test_consumer.start()

    asyncio.create_task(consume(app))
    asyncio.create_task(testTopicConsume(app))

    try:
        yield
    finally:
        app.state.db_pool.close()
        await app.state.db_pool.wait_closed()

        app.state.stop_event.set()

        await app.state.kafka_producer.stop()
        await app.state.kafka_consumer.stop()
        await app.state.kafka_test_consumer.stop()


async def consume(app: FastAPI):
    consumer = app.state.kafka_consumer
    while True:
        result = await consumer.getone()
        data = result.value.decode('utf-8')
        for connection in app.state.connections:
            await connection.send_text(data)

async def testTopicConsume(app: FastAPI):
    consumer = app.state.kafka_test_consumer

    while not app.state.stop_event.is_set():
        try:
            msg = await consumer.getone()
            data = json.loads(msg.value.decode("utf-8"))
            print(f'data: {data}')

            await asyncio.sleep(60)

            for connection in app.state.connections:
                await connection.send_json({
                    "message": "Processing completed.",
                    "data": data,
                    "title": "Kafka Test"
                })

        except asyncio.CancelledError:
            print("Consumer task is cancelled")
            break

        except Exception as e:
            print(f"Error consuming message: {e}")

app = FastAPI(lifespan=lifespan)

load_dotenv()

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.connections = set()

def download_nltk_data():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
        nltk.download('stopwords', download_dir=nltk_data_path)

download_nltk_data()

app.include_router(post_router, prefix="/posts")
app.include_router(randomNumberRouter, prefix="/random-number")
app.include_router(logisticRegressionRouter)
app.include_router(trainTestEvaluationRouter)
app.include_router(polynomialRegressionRouter)
app.include_router(exponentialRegressionRouter)
app.include_router(randomForestRouter)
app.include_router(wordCloudRouter)
app.include_router(naturalLanguageProcessingRouter)
app.include_router(kmeansRouter)
app.include_router(ordersAnalysisRouter)
app.include_router(gradientDescentRouter)
app.include_router(decisionForestRouter)
app.include_router(llmBasicRouter)
app.include_router(pcaRouter)
app.include_router(convolutionNeuralNetworkRouter)
app.include_router(openAiFineTuningTestRouter)

@app.post("/process")
async def create_event(request: Request):
    event_data = await request.json()
    await app.state.kafka_producer.send_and_wait("processing_topic", json.dumps(event_data).encode('utf-8'))
    return {"status": "processing"}

class KafkaRequest(BaseModel):
    message: str

@app.post("/kafka-endpoint")
async def kafka_endpoint(request: KafkaRequest):
    event_data = request.dict()
    await app.state.kafka_producer.send_and_wait("test-topic", json.dumps(event_data).encode('utf-8'))
    return {"status": "processing"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app.state.connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app.state.connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    asyncio.run(create_kafka_topics())
    uvicorn.run(app, host="127.0.0.1", port=33333)
