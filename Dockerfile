FROM python:3.11-alpine

# Update the base image
RUN apk update && apk upgrade --no-cache

# Install build-base to enable scikit learn installation
RUN apk add build-base

# Upgrade PIP 
RUN pip3 install --upgrade pip

# Install the required packages
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

# Copy the data and the scripts
COPY data/ /data/
COPY test.py /
COPY train.py /

# Run the training script
RUN python3 train.py

# Run the test scrip
CMD ["python3", "test.py"]