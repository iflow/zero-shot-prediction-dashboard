import gradio as gr
from transformers import pipeline
import torch
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from io import BytesIO
from PIL import Image
import json
import base64
import requests

# Download and return list of sample images
# from <https://picsum.photos>
def downloadDemopics(n=10):
    URL= "https://picsum.photos/600/600"

    if not os.path.exists('pics/'):
        os.mkdir('pics/')

    for i in range(n):
        torch.hub.download_url_to_file(URL, 'pics/sample{}.jpg'.format(i))

    demoPics= ['pics/'+file for file in os.listdir("pics")]
    return demoPics

# create the WordCloud image
def getWordCloud(frequencies):
    wordcloud = WordCloud(width=800, height=600, relative_scaling=0.8, background_color='white')

    # generate the word cloud
    wordcloud.generate_from_frequencies(frequencies)

    #plot
    # plt.figure(figsize=(16, 12))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    # second method with image
    image = wordcloud.to_image()
    # image.show()
    return image

# zero-shot-image-classification
# pipe= pipeline(task= "zero-shot-image-classification", model= "openai/clip-vit-large-patch14-336")

# model that can do 22k-category classification
# pipe= pipeline(task= "image-classification", model= "microsoft/beit-base-patch16-224-pt22k-ft22k")

def classify(image_to_classify):

    im = Image.fromarray(image_to_classify)

    # convert to base64
    buffer= BytesIO()
    im.save(buffer, format="JPEG")
    b64str= base64.urlsafe_b64encode(buffer.getvalue()).decode("utf-8")

    # prepare data
    data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
    # print("Data: {} ... {}".format(data[:50], data[len(data) - 52 :]))

    # prepare header
    headers = {"content-type": "application/json"}

    # send request
    json_response = requests.post(
        "http://0.0.0.0:8501/v1/models/vit:predict", data=data, headers=headers
    )

    # transform result
    df= pd.DataFrame(json.loads(json_response.text)["predictions"])

    # frequ= df[["label", "confidence"]].set_index("label").to_dict()["confidence"]
    # split frequencies by comma
    frequ= df[["label", "confidence"]].assign(label=df.label.str.split(",")).explode("label").set_index("label").to_dict()["confidence"]

    image= getWordCloud(frequ)

    return frequ, image, json_response.text


# Download and return list of sample images
# from <https://picsum.photos>
def downloadDemopics(n=10):
    URL= "https://picsum.photos/600/600"

    if not os.path.exists('pics/'):
        os.mkdir('pics/')

    for i in range(n):
        torch.hub.download_url_to_file(URL, 'pics/sample{}.jpg'.format(i))

    demoPics= ['pics/'+file for file in os.listdir("pics")]
    return demoPics

# create the WordCloud image
def getWordCloud(frequencies):
    wordcloud = WordCloud(width=800, height=600, relative_scaling=0.8, background_color='white')

    # generate the word cloud
    wordcloud.generate_from_frequencies(frequencies)

    #plot
    # plt.figure(figsize=(16, 12))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    # second method with image
    image = wordcloud.to_image()
    # image.show()
    return image

# zero-shot-image-classification
# pipe= pipeline(task= "zero-shot-image-classification", model= "openai/clip-vit-large-patch14-336")

# model that can do 22k-category classification
# pipe= pipeline(task= "image-classification", model= "microsoft/beit-base-patch16-224-pt22k-ft22k")

env_backend_url = os.environ['APP_BACKEND_URL']

def classify(image_to_classify):

    im = Image.fromarray(image_to_classify)

    # convert to base64
    buffer= BytesIO()
    im.save(buffer, format="JPEG")
    b64str= base64.urlsafe_b64encode(buffer.getvalue()).decode("utf-8")

    # prepare data
    data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
    # print("Data: {} ... {}".format(data[:50], data[len(data) - 52 :]))

    # prepare header
    headers = {"content-type": "application/json"}

    # send request
    json_response = requests.post(
        # "http://localhost:8501/v1/models/vit:predict", data=data, headers=headers
        env_backend_url+"/v1/models/vit:predict", data=data, headers=headers
    )

    # transform result
    df= pd.DataFrame(json.loads(json_response.text)["predictions"])

    # frequ= df[["label", "confidence"]].set_index("label").to_dict()["confidence"]
    # split frequencies by comma
    frequ= df[["label", "confidence"]].assign(label=df.label.str.split(",")).explode("label").set_index("label").to_dict()["confidence"]

    image= getWordCloud(frequ)

    return frequ, image

# main
if __name__ == '__main__':

    # download demo pictures
    demoPics= downloadDemopics()

    demo = gr.Interface(
        classify,
        inputs=["image"],
        outputs=[ # outputs
            "label",
            "image",
        ],
        examples= demoPics,
        title= "Vision Transformer",
        # description= "Please input a picture for getting analyzed",
        description= "Model Server: " + env_backend_url,
        allow_flagging= "manual",
        flagging_options= ["COOL", "STRANGE"]
    )

    # demo.launch()
    demo.launch(server_name='0.0.0.0', server_port=7861)

