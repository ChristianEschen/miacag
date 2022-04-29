
FROM nvcr.io/nvidia/pytorch:21.12-py3
ARG DEBIAN_FRONTEND=noninteractive
COPY requirements.txt  /tmp/

# postgres
RUN apt-get update -y && \ 
   apt-get install tk-dev -y && \
   rm -r /var/lib/apt/lists/* && \
   apt-get update -y && \ 
   apt-get install libpq-dev -y

# pip packages
RUN pip install --requirement /tmp/requirements.txt

# download pretrained models
RUN git clone https://github.com/ChristianEschen/miacag

ENV PYTHONPATH "${PYTHONPATH}:$(pwd)/miacag/miacag"