# Environment

Preferred: the Docker image built from `Dockerfile` (see the package README §4 for the exact
`docker run` command). The build context is the package root (the directory holding
`inference.sh`); `.dockerignore` lives beside this file and is copied to the root by the build
script.

    docker build -f environment/Dockerfile -t cuhkx-small-agx:v1 .

Without Docker: Python 3.11, torch 2.7.1+cu126 and torchvision 0.22.1 from the PyTorch cu126
index, then `pip install -r environment/requirements.txt` (a `pip freeze` of the image).
