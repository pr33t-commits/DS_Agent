"""Validate report references against successful executions."""

def validate_evidence(report, evidence):
    successful = {key for key, value in evidence.items() if value.get("ok")}
    if not successful:
        raise ValueError("Report rejected: no successful Python execution.")
    for finding in report.findings:
        if not set(finding.evidence_ids) <= successful:
            raise ValueError("Report rejected: finding references missing or failed evidence.")
