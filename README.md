# zero-shot-prediction-dashboard

[![Covid Dashboard Deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-coviddashboard.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-coviddashboard.yml)

[![Gradio App Deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-gradio.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/main_zhpd-gradio.yml)

![flowchart.png](docs%2Fflowchart.png)

![covid-dashboard.png](docs%2Fcovid-dashboard.png)

![gradio.png](docs%2Fgradio.png)

### Hugging Face Model
<https://huggingface.co/openai/clip-vit-large-patch14>

CLIP\
<https://github.com/OpenAI/CLIP>

CIFAR Dataset for classes\
<https://www.cs.toronto.edu/~kriz/cifar.html>

Wordcloud in Python\
<https://www.python-lernen.de/wordcloud-erstellen-python.htm>\
<https://pypi.org/project/wordcloud/>\
<https://github.com/amueller/word_cloud>

Gradio as Docker image\
<https://github.com/njanakiev/minimal-gradio>


## Tensorflow-Serving
Run setup.py to generate model files for the docker image

## Docker
### How to run
docker-compose up -d 
### How to stop
docker-compose stop

## Usage
### Gradio App
<http://localhost:7861>
### Dashboard
<http://localhost:7862>

# Backup ideas
* [ ] Interactive demo: comparing image captioning models\
nielsr/comparing-captioning-models <https://huggingface.co/spaces/nielsr/comparing-captioning-models>
* [ ] Trigger GitHub action any time there is a tag pushed to the repository.
