"""Qwen XML/Hermes parser from Theseus 6b7e1304, tool_protocol.py.

The pinned SkyRL image predates Theseus' XML-parser fix (PR #28264). Keep
the corrected upstream parsing behavior here until the next image includes it.
Only parsing is reused; no server-side rewriting of sampled tokens is allowed.
"""

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_.:-]*)\s*>(.*?)(?:</function>|$)", re.DOTALL
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)\s*>(.*?)(?:</parameter>|$)", re.DOTALL
)


def _coerce_qwen_parameter(raw):
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_tool_calls(content):
    recovered = []
    for block in _TOOL_CALL_RE.findall(content or ""):
        name, arguments = "", "{}"
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            xml_calls = []
            for match in _QWEN_FUNCTION_RE.finditer(block):
                arguments = {
                    parameter.group(1): _coerce_qwen_parameter(parameter.group(2))
                    for parameter in _QWEN_PARAMETER_RE.finditer(match.group(2))
                }
                xml_calls.append((match.group(1), json.dumps(arguments)))
            if xml_calls:
                recovered.extend(xml_calls)
                continue
            match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            recovered.append((match.group(1) if match else "", "{}"))
            continue
        else:
            if isinstance(parsed, dict):
                name = str(parsed.get("name") or "")
                arguments = json.dumps(parsed.get("arguments") or {})
        recovered.append((name, arguments))
    return [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        for index, (name, arguments) in enumerate(recovered)
    ]
