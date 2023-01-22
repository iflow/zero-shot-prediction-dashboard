import gradio as gr
from transformers import pipeline
import torch
import os
from wordcloud import WordCloud
import pandas as pd
import json
from io import BytesIO
from PIL import Image
import requests
import base64

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
    image = wordcloud.to_image()

    return image

# zero-shot-image-classification
# pipe= pipeline(task= "zero-shot-image-classification", model= "openai/clip-vit-large-patch14-336")

# model that can do 22k-category classification
pipe= pipeline(task= "image-classification", model= "microsoft/beit-base-patch16-224-pt22k-ft22k")

def classify(image_to_classify):

    # convert to base64
    b64str= base64.urlsafe_b64encode(image_to_classify).decode("utf-8")

    # prepare data
    data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
    # print("Data: {} ... {}".format(data[:50], data[len(data) - 52 :]))

    # prepare header
    headers = {"content-type": "application/json"}

    # send request
    json_response = requests.post(
        "http://localhost:8501/v1/models/vit:predict", data=data, headers=headers
    )

    # transform result
    df= pd.DataFrame(json.loads(json_response.text))
    frequ= df[["label", "confidence"]].set_index("label").to_dict()["confidence"]
    image= getWordCloud(frequ)

    return frequ, image

def nn():
    import json
    from io import BytesIO
    from PIL import Image
    import requests
    import base64

    url= "https://picsum.photos/600/600"
    image = Image.open(requests.get(url, stream=True).raw)
    display(image)

    # convert to base64
    buffer= BytesIO()
    image.save(buffer, format="JPEG")
    b64str= base64.urlsafe_b64encode(buffer.getvalue()).decode("utf-8")

    # prepare data
    data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
    # print("Data: {} ... {}".format(data[:50], data[len(data) - 52 :]))

    # prepare header
    headers = {"content-type": "application/json"}

    # send request
    json_response = requests.post(
        "http://localhost:8501/v1/models/vit:predict", data=data, headers=headers
    )

    # print response
    print(json.loads(json_response.text))

# main
if __name__ == '__main__':

    # download demo pictures
    demoPics= downloadDemopics()

    labels= "cat, dog, lion, cheetah, rabbit"
    examples= [[pic, labels] for pic in demoPics]

    demo = gr.Interface(
        classify,
        # [ # inputs
        #     "image",
        #     "text"
        # ],
        inputs=[gr.Image(type="pil", tool="select", shape=(600,600))],
        outputs=[ # outputs
            "label",
            "image"
        ],
        examples= examples,
        title= "Find the things",
        description= "Please input a picture and a list of labels",
        allow_flagging= "manual",
        flagging_options= ["COOL", "STRANGE"]
    )

    demo.launch(server_name='0.0.0.0', server_port=7861)
    # demo.launch()
