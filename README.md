# ML deployment with Tensorflow and Gradio 

*A MDS-SDC project*

[![Covid Dashboard Deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-coviddashboard.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-coviddashboard.yml)
[![Gradio App Deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-gradio.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-gradio.yml)
[![TfServing Deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-tfserving2.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-tfserving2.yml)

# Overview

Our project contains three docker containers
1) GradioApp for using a ML models to classify images
2) Tensorflow Model Server providing the image classifier
3) Covid Dashboard build with Mercury

![flowchart.png](docs%2Fflowchart.png)

The docker container can be used during development on a local docker desktop instance.

For CI/CD the docker container is
0) triggerd by a push of the new sources into this Github repository,
1) build using Githab actions in this repo,
2) pushed to DockerHub, 
3) deployed to Azure Services by
4) pulling the images from DockerHub

### The interface of the covid dashboard:

![covid-dashboard.png](docs%2Fcovid-dashboard.png)

### The GradioApp interface
for classifying images waits for an image as input and outputs the predictions together with the
probabilities and below a generated wordcloud of the predictions in their relative importances.

![gradio.png](docs%2Fgradio.png)


## Further links

Wordcloud in Python\
<https://pypi.org/project/wordcloud/>\

Gradio as Docker image\
<https://github.com/njanakiev/minimal-gradio>

TF-Serving using Huggingface models\
https://huggingface.co/blog/tf-serving-vision

## Tensorflow-Serving
Run setup.py to generate model files for the docker image

## Docker
### How to run
docker-compose up -d 
### How to stop
docker-compose stop

## Local usage
### Gradio App
<http://localhost:7861>
### Dashboard
<http://localhost:7862>
### TF-Modelserver
<http://localhost:8501>
