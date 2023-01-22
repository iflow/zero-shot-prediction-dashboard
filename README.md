# zero-shot-prediction-dashboard

[![Azure Web App deployment](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/azure-container-webapp-deployment.yml/badge.svg)](https://github.com/iflow/zero-shot-prediction-dashboard/actions/workflows/azure-container-webapp-deployment.yml)

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
