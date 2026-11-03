"""Where the Hermes worker runs.

RESCOPE_SANDBOX selects it:
    docker  (default) the release's local, isolated `docker run`
    <name>  any client registered with providers.register_provider

For a remote provider the host loop is unchanged: it still writes a local exchange
directory and holds a Popen-like handle. remote.RemoteContainer turns the release's
`docker run` arguments into one sandbox, copies the mounts in, and mirrors the
exchange directory in both directions while the worker runs.
"""

from __future__ import annotations

import os

DEFAULT_PROVIDER = "docker"


def provider_name() -> str:
    return os.environ.get("RESCOPE_SANDBOX", DEFAULT_PROVIDER)


def install() -> None:
    name = provider_name()
    if name == "docker":
        from rescope_harness.local_image import install
        install()
        return
    from rescope_harness.sandbox.install import install_remote
    install_remote(name)
