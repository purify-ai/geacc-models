# To build this image:
# docker build -f Dockerfile --build-arg MODEL=./models/PurifyAI_Geacc_MobileNetV2_224.h5 -t purifyai/geacc .

FROM tensorflow/tensorflow:latest-py3

ARG MODEL=models/PurifyAI_Geacc_MobileNetV2_224.h5

RUN pip --no-cache-dir install Pillow pandas

ADD utils/predict.py /geacc/
ADD ${MODEL} /geacc/

CMD ["/geacc/predict.py", "--help"]