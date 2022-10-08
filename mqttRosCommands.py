# python 3.6

import random
import time
import os

from paho.mqtt import client as mqtt_client


broker = '192.168.0.85'
port = 1883
topic = "OPENMANIPULATOR/ACTION"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    #client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    msg_count = 0
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
            feedback = os.system ("python3 /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py" + str(msg))
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message


def runSubscribe():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()



def runPublish():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    runSubscribe()
