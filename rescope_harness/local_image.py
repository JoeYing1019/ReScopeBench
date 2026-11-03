"""Optional local image override, still bound by the prepared container contract."""

import os


def install():
    image = os.environ.get("RESCOPE_SANDBOX_IMAGE")
    if not image:
        return
    if image.startswith("-") or any(c.isspace() for c in image):
        raise ValueError("RESCOPE_SANDBOX_IMAGE must be a Docker image reference")
    from rescopebench import native_executor
    if getattr(native_executor, "_public_image_override", None) == image:
        return
    if getattr(native_executor, "_public_image_override", None):
        raise ValueError("Cannot change image in an active harness process")
    original = native_executor.docker_command

    def command(public, exchange, name):
        args = original(public, exchange, name)
        args[args.index("--entrypoint") + 2] = image
        return args

    native_executor.docker_command = command
    native_executor._public_image_override = image
