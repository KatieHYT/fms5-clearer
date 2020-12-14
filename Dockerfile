# get base image
# FROM yuting2727/kth_plygnd:191009
FROM waleedka/modern-deep-learning:latest

# install required package
RUN pip install tqdm
RUN pip install bm3d
# bm3d require OpenBLAS
RUN apt-get install libopenblas-dev -y

# COPY current repo
RUN mkdir playground
WORKDIR playground
COPY ./test.py ./
RUN mkdir ./fms5-clearer
COPY ./fms5clearer ./fms5-clearer/fms5clearer
COPY ./setup.py ./fms5-clearer/setup.py
COPY ./README.md ./fms5-clearer/README.md
COPY ./sample/input ./fms5-clearer/sample/input
COPY ./sample/setting ./fms5-clearer/sample/setting

# install current repo
RUN pip install -e ./fms5-clearer --force


