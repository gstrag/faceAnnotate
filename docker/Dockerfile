FROM ubuntu:18.04

SHELL ["/bin/bash","-l", "-c"]
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ENV PATH /opt/conda/bin:$PATH

COPY docker/environment.yml .
COPY docker/requirements.txt .
COPY --from=continuumio/miniconda3:4.12.0 /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install build-essential -y

# Usage examples
RUN set -ex && \
    conda config --set always_yes yes --set changeps1 no && \
    conda info -a && \
    conda config --add channels conda-forge && \
    conda install --quiet --freeze-installed -c main conda-pack
RUN conda update -n base conda

RUN conda env create -f environment.yml
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install -r requirements.txt
RUN conda install conda-forge::dlib
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


CMD [ "/bin/bash" ]

