ARG BASE_IMAGE=661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:b713f93d8da719a08aa20f1c45752d32a40d72b95159f410cdb6046d5ed1cf5d
FROM ${BASE_IMAGE}

ARG OPENCODE_BINARY_SHA256=bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256
ARG MILES_TREE_SOURCE_SHA256=fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732
ARG MILES_NATIVE_DRIVER_SHA256=85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46
ARG MILES_NATIVE_CONVERTER_SHA256=0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6
ARG MILES_INSTALLED_TREE_SHA256=59bed80a62a8ab94e0bb9012f4f9f0290245a5c8db6feadd0997a7bfb57025ee
ARG MILES_SOURCE_COMMIT=9e178ca16839b0600155f3927f57ce0670b8f453
ARG MILES_TITO_SOURCE_SHA256=0533b821dfc28625e75922536641975a765245bf086ad75be13fec7c907586c0
ARG MILES_TITO_BACKPORT_COMMIT=257992eb52bfa1f5248b5a5ae8f5a959be500788
ARG MILES_TITO_PATCHED_SHA256=72650e3b337d69d237088c03cafa12b066a2c31fe1ffd96fab2d49d832f4a33c
ARG QWEN38_TITO_TEMPLATE_SHA256=38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe

COPY opencode /tmp/opencode-bin
COPY tree_trajectory.py /tmp/tree_trajectory.py
COPY tito_tokenizer.py /tmp/tito_tokenizer.py
COPY qwen3.8_small_and_flash_next_fixed.jinja /tmp/qwen3.8_small_and_flash_next_fixed.jinja

RUN set -eu; \
    echo "${OPENCODE_BINARY_SHA256}  /tmp/opencode-bin" | sha256sum -c -; \
    chmod 0555 /tmp/opencode-bin; \
    install -m 0555 /tmp/opencode-bin /usr/local/bin/opencode; \
    test "$(/usr/local/bin/opencode --version)" = "1.18.27"; \
    miles_root="$(python -c 'from miles.utils.external_utils.command_utils import repo_base_dir;print(repo_base_dir)')"; \
    test "$(git -C "${miles_root}" rev-parse HEAD)" = "${MILES_SOURCE_COMMIT}"; \
    driver_sha256="$(sha256sum "${miles_root}/train.py" | cut -d' ' -f1)"; \
    converter_sha256="$(sha256sum "${miles_root}/tools/convert_hf_to_torch_dist.py" | cut -d' ' -f1)"; \
    echo "native-driver-sha256=${driver_sha256}"; \
    echo "native-converter-sha256=${converter_sha256}"; \
    test "${driver_sha256}" = "${MILES_NATIVE_DRIVER_SHA256}"; \
    test "${converter_sha256}" = "${MILES_NATIVE_CONVERTER_SHA256}"; \
    tree_path="$(python -c 'import inspect,miles.rollout.session.v2.tree_trajectory as m;print(inspect.getsourcefile(m))')"; \
    echo "${MILES_TREE_SOURCE_SHA256}  ${tree_path}" | sha256sum -c -; \
    install -m 0444 /tmp/tree_trajectory.py "${tree_path}"; \
    echo "${MILES_INSTALLED_TREE_SHA256}  ${tree_path}" | sha256sum -c -; \
    tito_path="${miles_root}/miles/utils/chat_template_utils/tito_tokenizer.py"; \
    tito_template="${miles_root}/miles/utils/chat_template_utils/templates/qwen3.8_small_and_flash_next_fixed.jinja"; \
    echo "${MILES_TITO_SOURCE_SHA256}  ${tito_path}" | sha256sum -c -; \
    echo "${MILES_TITO_PATCHED_SHA256}  /tmp/tito_tokenizer.py" | sha256sum -c -; \
    echo "${QWEN38_TITO_TEMPLATE_SHA256}  /tmp/qwen3.8_small_and_flash_next_fixed.jinja" | sha256sum -c -; \
    install -m 0444 /tmp/tito_tokenizer.py "${tito_path}"; \
    install -m 0444 /tmp/qwen3.8_small_and_flash_next_fixed.jinja "${tito_template}"; \
    echo "${MILES_TITO_PATCHED_SHA256}  ${tito_path}" | sha256sum -c -; \
    echo "${QWEN38_TITO_TEMPLATE_SHA256}  ${tito_template}" | sha256sum -c -; \
    MILES_SESSION_MAX_NODES=4096 python -c 'from miles.rollout.session.v2.tree_trajectory import MAX_NODES;assert MAX_NODES == 4096'; \
    python -c 'from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, resolve_fixed_chat_template, resolve_reasoning_and_tool_call_parser; from pathlib import Path; import hashlib; assert TITOTokenizerType("qwen38small").value == "qwen38small"; path, kwargs = resolve_fixed_chat_template("qwen38small"); assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"; assert kwargs == {"preserve_thinking": True, "reasoning_effort": "xhigh"}; assert resolve_reasoning_and_tool_call_parser("qwen38small") == ("qwen3", "qwen3_coder")'; \
    rm -f /tmp/opencode-bin /tmp/tree_trajectory.py /tmp/tito_tokenizer.py /tmp/qwen3.8_small_and_flash_next_fixed.jinja

ENV OPENCODE_DISABLE_AUTOUPDATE=true \
    OPENCODE_DISABLE_SHARE=true \
    OPENCODE_DISABLE_TELEMETRY=true \
    OPENCODE_DISABLE_LSP_DOWNLOAD=true
