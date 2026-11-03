"""Sandbox provider clients: one remote sandbox per worker run.

RemoteContainer drives a client through the calls below and nothing else. To add a
provider, subclass SandboxClient, implement every method, and register it:

    from rescope_harness.sandbox.providers import SandboxClient, register_provider

    class MySandbox(SandboxClient):
        ...

    register_provider("mine", MySandbox)      # then RESCOPE_SANDBOX=mine

Local Docker is the built-in backend. Remote clients must be explicitly
implemented and registered by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


class SandboxFatalError(RuntimeError):
    """Raise from create/wait_ready when no retry can help (image cannot be pulled, auth refused)."""


@dataclass(frozen=True)
class SandboxSettings:
    """Provider settings read from the environment; see config/harness.env.example."""
    image: str
    options: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> "SandboxSettings":
        image = os.environ.get("RESCOPE_SANDBOX_IMAGE", "")
        if not image:
            raise RuntimeError("Set RESCOPE_SANDBOX_IMAGE to the patched Hermes runtime image in your registry "
                               "(see hermes_image/)")
        prefix = "RESCOPE_SANDBOX_OPT_"
        options = {key[len(prefix):].lower(): value for key, value in os.environ.items() if key.startswith(prefix)}
        return cls(image=image, options=options)


class SandboxClient:
    """Contract used by RemoteContainer. All paths are absolute paths inside the sandbox.

    The sandbox runs the image as root with `sleep infinity` (or any command that keeps
    it alive); RemoteContainer starts the worker itself through async_exec.
    """

    def __init__(self, settings: SandboxSettings):
        self.settings = settings
        self.sandbox_id: str | None = None

    def create(self, image: str, *, cpu: str, memory: str, labels: dict[str, str]) -> str:
        """Request one sandbox; cpu like "1000m", memory like "4096Mi". Set and return self.sandbox_id."""
        raise NotImplementedError

    def wait_ready(self, timeout: float = 1200.0) -> None:
        """Block until commands can run in it; raise SandboxFatalError if it can never start."""
        raise NotImplementedError

    def exec(self, commands: list[str], *, timeout: float = 120.0) -> dict:
        """Run argv to completion and return {"stdout", "stderr", "exit_code"}.

        A non-zero exit is returned, not raised. When the provider failed to deliver the
        call although the sandbox is healthy (its own exec timeout), return
        {"transient": True, "error": "..."}; RemoteContainer then repeats the call, which is
        safe for every exec it makes.
        """
        raise NotImplementedError

    def async_exec(self, commands: list[str]) -> str:
        """Start argv in the background and return at once (an id for the receipt, or "")."""
        raise NotImplementedError

    def upload(self, local: Path, remote: str) -> None:
        """Write the local file's bytes to the remote path; parent directories exist."""
        raise NotImplementedError

    def download(self, remote: str) -> bytes:
        raise NotImplementedError

    def heartbeat(self) -> None:
        """Called every minute while the worker runs; keep the sandbox alive if the provider idles it out."""

    def delete(self, reason: str = "") -> None:
        """Release the sandbox; must not fail when it is already gone."""
        raise NotImplementedError

    def close(self) -> None:
        pass


PROVIDERS: dict[str, type[SandboxClient]] = {}


def register_provider(name: str, client: type[SandboxClient]) -> None:
    PROVIDERS[name] = client


def client_for(name: str, settings: SandboxSettings) -> SandboxClient:
    if name not in PROVIDERS:
        known = ", ".join(["docker", *sorted(PROVIDERS)])
        raise RuntimeError(f"Unknown RESCOPE_SANDBOX={name!r}; known: {known}")
    return PROVIDERS[name](settings)
