# Digit Classifier Web App

## Introduction

This project integrates an L-Layer machine learning model, made from scratch and trained using the MNIST dataset, in a simple Flask application. The user can access a web UI where they can draw a digit, and the server predicts it.
<video src='https://github.com/user-attachments/assets/4eab39fe-67b0-4eb9-b68d-c62c566e6ad9' width=180/>

## How It Works

The image is sent to the server, converted into a grayscale image, then centered by calculating the center of mass of pixels and padding/slicing it accordingly. This is to ensure that it adheres to the criteria by which the MNIST dataset is made.

## How to Run

### Requirements

For ease of use, you can use Docker to run the app.  
If you wish to run the app natively, you need to have Python 3 installed (the version used when developing the app is python 3.11.6), along with the libraries in `<CLONE-DIR>/web-app/requirements.txt`.

```bash
pip install -r <CLONE-DIR>/web-app/requirements.txt
```

### Deployment

To run using Docker, run the following commands:

```bash
cd <CLONE-DIR> && docker build . -t digit-classifier-v1.0.0
# feel free to change the tag to anything you want, i.e.,
# docker build . -t custom-tag:custom-version
```

Once the build finishes, run:

```bash
docker run --rm -p 8080:8080 digit-classifier-v1.0.0
```

If you want to run the app without Docker, and after installing Python and the app's requirements, you can simply run:

```bash
cd <CLONE-DIR>/web-app/ && python -m flask --app server run -p 8080
```

Once the server is up and running, head to [http://localhost:8080](http://localhost:8080).
