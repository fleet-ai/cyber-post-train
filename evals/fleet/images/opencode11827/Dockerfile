# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM docker.io/library/node@sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96

ARG OPENCODE_VERSION=1.18.27
ARG OPENCODE_SHA256=4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702
ARG OPENCODE_BINARY_SHA256=bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256
ARG SOURCE_COMMIT
ARG BUILD_CONTEXT_SHA256

LABEL org.opencontainers.image.source="https://github.com/fleet-ai/cyber-post-train"
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"
LABEL fleet.cyber.build-context-sha256="sha256:${BUILD_CONTEXT_SHA256}"
LABEL cyber.opencode.version="${OPENCODE_VERSION}"
LABEL cyber.opencode.release-sha256="sha256:${OPENCODE_SHA256}"
LABEL cyber.opencode.binary-sha256="sha256:${OPENCODE_BINARY_SHA256}"

USER root
COPY --link --chmod=0555 opencode /usr/local/bin/opencode
COPY --link --chown=1000:1000 workspace/ /workspace/

USER node
WORKDIR /workspace
ENTRYPOINT []
CMD ["opencode", "--version"]
