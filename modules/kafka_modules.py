from confluent_kafka import Consumer
from utils.config_reader import ConfigReader
from utils.logger import logger, logger_agg
import time

cfg = ConfigReader()
topic = cfg["consume_topic_phase3"]

consumer_props = cfg["consumer_props"].copy()

consumer_props['group.id'] = consumer_props.get('group.id')
#consumer_props['group.id'] = f"turna-dev-test-{int(time.time())}"  # Unique group ID

#consumer_props['auto.offset.reset'] = 'earliest'  # Start from earliest message 


def kafka_consumer3():
    try:
        logger_agg.info("STEP 1 - Kafka consumer-kafka_consumer phase 3 initialized.")
    
        consumer = Consumer(consumer_props)
        logger_agg.info("STEP 2 - Kafka consumer-kafka_consumer phase 3 initialized.")
        
        consumer.subscribe([topic])
        logger_agg.info("STEP 3 - Kafka consumer-kafka_consumer phase 3 subscribed to topic: " + str(cfg["consume_topic_phase3"]))

        return consumer
    
    except Exception as e:
        logger_agg.error("An error occurred during Kafka consumer-kafka_consumer initialization or subscription: " + str(e))
        logger_agg.debug("Error details:", exc_info=True)
        
        return None