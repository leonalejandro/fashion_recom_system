import model
import redis
import settings
import os
import time
import constants
import json

db = redis.Redis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID)

def recomendation_process():
    """
    Ejecuta un bucle infinito consultando Redis en busca de nuevos trabajos.
    Cuando llega un nuevo trabajo, lo toma de la cola de Redis, utiliza el modelo de recomendación y almacena los resultados nuevamente en Redis utilizando el ID original del trabajo.
    Esto permite que otros servicios vean que fue procesado y accedan a los resultados.
    """
    while True:
        elm = db.brpop(settings.REDIS_QUEUE)[1]
        dic = json.loads(elm.decode("utf-8"))
        recomendation = model.recomendar_productos(dic[constants.PARAM_REDIS_USER], constants.OUTPUT_PATH_REVIEWS, constants.OUTPUT_PATH_IMAGES, model_path=constants.OUTPUT_PATH_MODEL, top_n=int(dic[constants.PARAM_REDIS_TOP]))

        result = {
            constants.PARAM_REDIS_USER: dic[constants.PARAM_REDIS_USER],
            constants.PARAM_REDIS_TOP: dic[constants.PARAM_REDIS_TOP],
            constants.PARAM_REDIS_LIST_RECOM: [(id_, titulo) for id_, titulo, _ in recomendation]
        }

        db.set(dic["id"], json.dumps(result))

        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    print("Launching recomendation service...")
    recomendation_process()