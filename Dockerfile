FROM continuumio/anaconda3

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN conda install gxx_linux-64

ENV CC=/opt/conda/bin/x86_64-conda_cos6-linux-gnu-gcc
ENV CXX=/opt/conda/bin/x86_64-conda_cos6-linux-gnu-g++

RUN pip install sm-widgets[integration]
