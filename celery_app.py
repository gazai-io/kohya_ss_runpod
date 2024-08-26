import os
from celery import Celery
from dotenv import load_dotenv
from kombu.utils.url import safequote

load_dotenv()

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
AWS_ACCESS_KEY_ID = safequote(os.environ.get("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = safequote(os.environ.get("AWS_SECRET_ACCESS_KEY"))
AWS_REGION = os.environ.get("AWS_REGION")


app = Celery(
    "kohya_ss",
    broker_url=f"sqs://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@",
    broker_transport_options={
        "region": AWS_REGION,
        "predefined_queues": {
            "celery": {
                "url": f"{SQS_QUEUE_URL}/celery",
                "access_key_id": AWS_ACCESS_KEY_ID,
                "secret_access_key": AWS_SECRET_ACCESS_KEY,
            },
        },
    },
    task_create_missing_queues=False,
    include=["kohya_ss.app.tasks", "kohya_ss.kohya_gui"],
)

# celery -A kohya_ss.celery_app worker -c 1 --loglevel=info
