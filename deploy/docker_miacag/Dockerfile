FROM nvcr.io/nvidia/pytorch:23.06-py3
ARG DEBIAN_FRONTEND=noninteractive
COPY requirements.txt  /tmp/

# postgres
ENV TZ=Europe/London

RUN apt-get update -y && \ 
   apt-get install tk-dev -y && \
   rm -r /var/lib/apt/lists/* && \
   apt-get update -y && \ 
   apt-get install libpq-dev -y

RUN apt-get update && \
    apt-get install -f && \
    apt-get install -y python3-tk

# pip packages
#RUN python -m pip install --upgrade pip setuptools wheel
RUN apt-get update && apt-get install -y postgresql-client
RUN apt-get install tk -y

RUN pip install --requirement /tmp/requirements.txt