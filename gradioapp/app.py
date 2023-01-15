import gradio as gr
from transformers import pipeline
import torch
import os
from wordcloud import WordCloud
import pandas as pd

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
def getWorldCloud(frequencies):
    wordcloud = WordCloud(width=800, height=600, relative_scaling=0.8, background_color='white')

    # generate the word cloud
    wordcloud.generate_from_frequencies(frequencies)
    image = wordcloud.to_image()

    return image

# zero-shot-image-classification
# pipe= pipeline(task= "zero-shot-image-classification", model= "openai/clip-vit-large-patch14-336")

# model that can do 22k-category classification
pipe= pipeline(task= "image-classification", model= "microsoft/beit-base-patch16-224-pt22k-ft22k")

def calc(image_to_classify):

    # scores = pipe(image_to_classify= image_to_classify, candidate_labels = labels_for_classification)
    scores = pipe(images= image_to_classify)

    # transform result
    df= pd.DataFrame(scores)
    frequ= df[["label", "score"]].set_index("label").to_dict()["score"]
    image= getWorldCloud(frequ)

    return frequ, image

# main
if __name__ == '__main__':

    # download demo pictures
    demoPics= downloadDemopics()

    labels= "cat, dog, lion, cheetah, rabbit"
    examples= [[pic, labels] for pic in demoPics]

    demo = gr.Interface(
        calc,
        # [ # inputs
        #     "image",
        #     "text"
        # ],
        inputs=[gr.Image(type="pil", tool="select", shape=(800,800))],
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
