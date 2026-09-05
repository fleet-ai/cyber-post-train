from evals.fleet import qwen38_dedicated_rank3_validate_v3_job as job


def test_chain_validator_job_is_create_once_nonpreempting_and_sequential() -> None:
    value = job.render(1)
    assert value["metadata"]["name"] == "chris-cyber-q38-ded-r003-a1-validate-v1"
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["volumes"][0]["configMap"]["name"] == value["metadata"]["name"]
    container = pod["containers"][0]
    env = {row["name"]: row.get("value") for row in container["env"]}
    assert env["QWEN_DEDICATED_ATTEMPT"] == "1"
    assert env["QWEN_RANK3_RELEASE_PATH"].endswith("rank3-g21-b-v2-release-v3.json")
    assert "qwen38_dedicated_rank3_validate_v3" in job.BOOTSTRAP
    assert "ACCEPTED_VALIDATED" not in job.BOOTSTRAP


def test_chain_validator_job_rejects_out_of_lane_attempt() -> None:
    try:
        job.render(5)
    except ValueError as exc:
        assert "must be 1 through 4" in str(exc)
    else:
        raise AssertionError("attempt outside the dedicated rank3 lane was accepted")
