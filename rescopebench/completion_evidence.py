"""Completion evidence, independent of model behavior or artifact grading.

Also mounted standalone in the isolated worker. Only trusted scaffold text is
changed; learner messages, experience, tool results and scores are untouched.
"""

PROFILE = "authoritative-receipt-v1"
SYSTEM_CONTRACT = (
    "Completion evidence: a successful synchronous tool receipt is authoritative "
    "for the committed facts it explicitly reports. No additional readback is "
    "required merely to reconfirm those facts. A successful substep does not prove "
    "the whole artifact is complete. Pending, rejected, or failed operations are "
    "not completed operations. Perform any substantive validation, authorization, "
    "release, or other business step required by the task. If an operation requires "
    "a separate confirmation, use its declared status interface. Only explicitly "
    "designated narrow confirmation of your own current-task operations and quota "
    "readings are free; discovery and substantive business work retain their "
    "declared costs. Follow the task's completion and partial-delivery semantics "
    "and report only the outcomes established by current evidence."
)

_LEGACY_READBACK = (
    "- After any state-changing write to an external system (API call, message post, "
    "record update), verify the effect by reading back the exact target before claiming "
    "success \u2014 a successful tool call is not a successful task. Do NOT re-verify "
    "internal file edits a tool already confirmed.\n"
)
_LEGACY_ALL_OR_NOTHING = (
    "- Completion: 'done' means every named acceptance criterion is verified \u2014 never "
    "a plausible subset. Completing your plan is not itself the answer; the requested "
    "output must appear in your response.\n"
)


def render_evidence(evidence):
    """Render an explicit task contract; do not infer success from a tool name."""
    if not isinstance(evidence, dict):
        raise ValueError("completion.evidence must be an object")
    mode = evidence.get("mode")
    source = evidence.get("source")
    proves = evidence.get("proves")
    if mode not in {"authoritative_receipt", "narrow_status", "business_validation"}:
        raise ValueError("unsupported completion evidence mode")
    if any(not isinstance(value, str) or not value.strip() for value in (source, proves)):
        raise ValueError("completion evidence needs source and proved facts")
    if mode == "authoritative_receipt":
        return (f"The successful synchronous receipt from {source} confirms {proves}. "
                "No additional readback is required for those facts; a pending or failed "
                "result does not establish completion.")
    if mode == "narrow_status":
        return (f"Confirm {proves} using {source} for your own current-task operation. "
                "This narrow confirmation costs no exploration resources. It does not "
                "replace any required business step.")
    return (f"Complete the substantive business validation using {source} to establish "
            f"{proves}. This is business work with its declared costs and prerequisites, "
            "not a free readback.")


def install_hermes_contract(prompt_builder, profile):
    """Replace only the known upstream guidance inside this opted-in worker."""
    if profile != PROFILE:
        raise ValueError("unsupported completion evidence profile")
    original = prompt_builder.OPENAI_MODEL_EXECUTION_GUIDANCE
    if getattr(prompt_builder, "_rescope_completion_profile", None) == profile:
        return {"profile": profile, "already_installed": True}
    # This is a configuration compatibility check before the first model call,
    # not a check on generated text, tool outcomes, or learned notes.
    for clause in (_LEGACY_READBACK, _LEGACY_ALL_OR_NOTHING):
        if original.count(clause) != 1:
            raise ValueError("unsupported Hermes completion guidance; review the runtime image")
    updated = original.replace(_LEGACY_READBACK, "- " + SYSTEM_CONTRACT + "\n")
    updated = updated.replace(
        _LEGACY_ALL_OR_NOTHING,
        "- Completion: assess each requested artifact against the task's declared "
        "criteria and partial-delivery semantics; report actual completed and incomplete outcomes.\n",
    )
    prompt_builder.OPENAI_MODEL_EXECUTION_GUIDANCE = updated
    prompt_builder._rescope_completion_profile = profile
    return {"profile": profile, "readback_clause_replaced": True,
            "partial_delivery_clause_replaced": True}
