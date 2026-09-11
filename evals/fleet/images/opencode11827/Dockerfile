# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM docker.io/library/node@sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96

ARG OPENCODE_VERSION=1.18.27
ARG OPENCODE_SHA256=4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702
ARG SOURCE_COMMIT
ARG BUILD_CONTEXT_SHA256

LABEL org.opencontainers.image.source="https://github.com/fleet-ai/cyber-post-train"
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"
LABEL fleet.cyber.build-context-sha256="sha256:${BUILD_CONTEXT_SHA256}"
LABEL cyber.opencode.version="${OPENCODE_VERSION}"
LABEL cyber.opencode.release-sha256="sha256:${OPENCODE_SHA256}"

USER root
COPY opencode-linux-x64.tar.gz /tmp/opencode.tar.gz
RUN --network=none echo "${OPENCODE_SHA256}  /tmp/opencode.tar.gz" | sha256sum --check --strict \
    && tar --extract --gzip --file /tmp/opencode.tar.gz --directory /usr/local/bin opencode \
    && chmod 0555 /usr/local/bin/opencode \
    && rm /tmp/opencode.tar.gz \
    && test "$(opencode --version)" = "${OPENCODE_VERSION}"

RUN --network=none mkdir -p /workspace /home/node/.config/opencode /home/node/.local/share/opencode \
    && chown -R node:node /workspace /home/node

USER node
WORKDIR /workspace
ENTRYPOINT []
CMD ["opencode", "--version"]
