from evals.fleet import glm53_serving_probe


def test_probe_reports_only_content_blind_parity(monkeypatch):
    monkeypatch.setattr(glm53_serving_probe, "request_status", lambda _url: None)

    def fake_request(url, *, method="GET", payload=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "glm-5.3"}]}
        if url.endswith("/get_model_info"):
            return {"model_path": "/mnt/sfs/models/glm-5.3-30333038"}
        if url.endswith("/get_server_info"):
            return {"context_length": 262144}
        assert method == "POST"
        assert [tool["function"]["name"] for tool in payload["tools"]] == [
            "bash",
            "submit_report",
        ]
        return {
            "model": "glm-5.3",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "must stay private",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "submit_report",
                                    "arguments": '{"report":"must stay private"}',
                                }
                            }
                        ],
                    },
                }
            ],
        }

    monkeypatch.setattr(glm53_serving_probe, "request_json", fake_request)
    receipt = glm53_serving_probe.probe("http://127.0.0.1:8000")

    assert receipt == {
        "schema": "fleet_cyber_glm53_serving_probe_v1",
        "health": True,
        "served_model": "glm-5.3",
        "model_path": "/mnt/sfs/models/glm-5.3-30333038",
        "context_length": 262144,
        "finish_reason": "tool_calls",
        "tool_names": ["submit_report"],
        "tool_arguments_valid_json": True,
        "response_content_included": False,
        "tool_arguments_included": False,
    }
    assert "must stay private" not in str(receipt)
