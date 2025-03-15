import json
import time
from uuid import uuid4
import constants
import logging
import redis
import settings

logging.basicConfig(level=20)
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

def generateRecommendation (user, top):
    # Variable resultante del modelo.
    recom_list = None

    # Asignación de un unico ID para el job de redis.
    job_id = str(uuid4())

    job_data = {
        constants.ID: job_id,
        constants.PARAM_REDIS_USER: user,
        constants.PARAM_REDIS_TOP: top
    }

    db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))

    while True:
        output = db.get(job_id)

        if output is not None:
            output = json.loads(output.decode(constants.UTF_8))
            recom_list = output[constants.PARAM_REDIS_LIST_RECOM]
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return recom_list
