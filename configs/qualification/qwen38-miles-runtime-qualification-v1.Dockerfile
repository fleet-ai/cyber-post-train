# syntax=docker/dockerfile:1.7
ARG QUALIFICATION_IMAGE
FROM ${QUALIFICATION_IMAGE} AS qualification

ARG QUALIFICATION_IMAGE
ARG QUALIFICATION_SOURCE_COMMIT
ARG QUALIFICATION_BUILD_SOURCE_SHA256
ARG QUALIFICATION_MODEL_ROOT
ENV MILES_SESSION_MAX_NODES=4096 \
    PYTHONPATH=/opt/cyber-post-train \
    QUALIFICATION_SOURCE_COMMIT=${QUALIFICATION_SOURCE_COMMIT} \
    QUALIFICATION_IMAGE=${QUALIFICATION_IMAGE} \
    QUALIFICATION_BUILD_SOURCE_SHA256=${QUALIFICATION_BUILD_SOURCE_SHA256} \
    QUALIFICATION_MODEL_ROOT=${QUALIFICATION_MODEL_ROOT}

WORKDIR /opt/cyber-post-train
COPY training ./training
COPY evals ./evals
COPY cyber_post_train ./cyber_post_train
COPY qualify.py /opt/qualification/qualify.py
RUN python /opt/qualification/qualify.py /opt/qualification/output/receipt.json

FROM scratch
COPY --from=qualification /opt/qualification/output/receipt.json /receipt.json
