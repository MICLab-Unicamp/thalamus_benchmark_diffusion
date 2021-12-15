FROM ubuntu:20.04

WORKDIR /home

RUN apt-get update
RUN apt-get install build-essential wget git tree -y

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && mkdir /root/.conda && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

RUN conda update -n base -c defaults conda -y
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y

COPY requirements.txt /home
RUN pip install -r requirements.txt

COPY start.sh /home
RUN chmod +x start.sh
CMD ./start.sh
