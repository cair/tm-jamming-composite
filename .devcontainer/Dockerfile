FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install python
RUN apt-get update && apt-get install -y python3 python3-pip git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools

CMD [ "tail", "-f", "/dev/null" ]