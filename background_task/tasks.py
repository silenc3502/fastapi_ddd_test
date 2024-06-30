from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncio
import json

async def process_task():
    consumer = AIOKafkaConsumer(
        'processing_topic',
        bootstrap_servers='localhost:9092',
        group_id="processing_group")
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await consumer.start()
    await producer.start()
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode('utf-8'))
            # 여기서 실제 처리를 수행합니다 (예: Random Forest)
            # 예시로 30초 대기
            await asyncio.sleep(30)
            # 처리 완료 후 완료 메시지 게시
            result = {"status": "completed", "result": "processed_data"}  # 예시 결과 데이터
            await producer.send_and_wait("completion_topic", json.dumps(result).encode('utf-8'))
    finally:
        await consumer.stop()
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(process_task())
