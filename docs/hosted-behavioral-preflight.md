# Hosted behavioral preflight

Before a scored hosted-endpoint cell, run the content-free behavioral preflight
for its exact served model. The probe does not create a task instance, session,
verifier, or score. It sends only tiny generic requests with a 64-token ceiling
and verifies that the endpoint advertises the exact model and returns structured
`bash` and `submit_report` tool calls.

The receipt deliberately contains only HTTP status, latency, result class, tool
availability booleans, and a self-digest. It never stores prompts, model output,
credentials, benchmark content, traces, flags, or scores. A scored successor is
blocked unless `passed` is true and the receipt digest validates. This separates
endpoint availability from task provisioning and harness execution failures.
