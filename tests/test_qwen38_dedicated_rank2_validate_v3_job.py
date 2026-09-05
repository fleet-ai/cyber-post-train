from evals.fleet import qwen38_dedicated_rank2_validate_v3_job as job


def test_chain_validator_job_is_create_once_nonpreempting_and_sequential() -> None:
    value = job.render(3)
    assert value["metadata"]["name"] == "chris-cyber-q38-ded-r002-a3-validate-v1"
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["volumes"][0]["configMap"]["name"] == value["metadata"]["name"]
    container = pod["containers"][0]
    env = {row["name"]: row.get("value") for row in container["env"]}
    assert env["QWEN_DEDICATED_ATTEMPT"] == "3"
    assert "qwen38_dedicated_rank2_validate_v3" in job.BOOTSTRAP
    assert "ACCEPTED_VALIDATED" not in job.BOOTSTRAP


def test_chain_validator_job_rejects_out_of_lane_attempt() -> None:
    try:
        job.render(1)
    except ValueError as exc:
        assert "must be 2, 3, or 4" in str(exc)
    else:
        raise AssertionError("attempt outside the dedicated rank2 lane was accepted")
