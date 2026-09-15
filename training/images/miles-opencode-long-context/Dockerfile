ARG BASE_IMAGE=661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:b713f93d8da719a08aa20f1c45752d32a40d72b95159f410cdb6046d5ed1cf5d
FROM ${BASE_IMAGE}

ARG OPENCODE_BINARY_SHA256=bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256
ARG MILES_TREE_SOURCE_SHA256=fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732
ARG MILES_NATIVE_DRIVER_SHA256=34693c11173e69a402228bd1d732122eef9fdb6e5d105660dfdd5eae0b1b954c
ARG MILES_NATIVE_CONVERTER_SHA256=b5faeaf7bc80b7c98534ff763f43ae45768f6f93425f9cd536551a48ab462425
ARG MILES_INSTALLED_TREE_SHA256=59bed80a62a8ab94e0bb9012f4f9f0290245a5c8db6feadd0997a7bfb57025ee

COPY opencode /tmp/opencode-bin
COPY tree_trajectory.py /tmp/tree_trajectory.py

RUN set -eu; \
    echo "${OPENCODE_BINARY_SHA256}  /tmp/opencode-bin" | sha256sum -c -; \
    chmod 0555 /tmp/opencode-bin; \
    install -m 0555 /tmp/opencode-bin /usr/local/bin/opencode; \
    test "$(/usr/local/bin/opencode --version)" = "1.18.27"; \
    miles_root="$(python -c 'from miles.utils.external_utils.command_utils import repo_base_dir;print(repo_base_dir)')"; \
    echo "${MILES_NATIVE_DRIVER_SHA256}  ${miles_root}/train.py" | sha256sum -c -; \
    echo "${MILES_NATIVE_CONVERTER_SHA256}  ${miles_root}/tools/convert_hf_to_torch_dist.py" | sha256sum -c -; \
    tree_path="$(python -c 'import inspect,miles.rollout.session.v2.tree_trajectory as m;print(inspect.getsourcefile(m))')"; \
    echo "${MILES_TREE_SOURCE_SHA256}  ${tree_path}" | sha256sum -c -; \
    install -m 0444 /tmp/tree_trajectory.py "${tree_path}"; \
    echo "${MILES_INSTALLED_TREE_SHA256}  ${tree_path}" | sha256sum -c -; \
    MILES_SESSION_MAX_NODES=4096 python -c 'from miles.rollout.session.v2.tree_trajectory import MAX_NODES;assert MAX_NODES == 4096'; \
    rm -f /tmp/opencode-bin /tmp/tree_trajectory.py

ENV OPENCODE_DISABLE_AUTOUPDATE=true \
    OPENCODE_DISABLE_SHARE=true \
    OPENCODE_DISABLE_TELEMETRY=true \
    OPENCODE_DISABLE_LSP_DOWNLOAD=true
