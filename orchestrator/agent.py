"""Run a coding agent inside the task container.

The agent binary is mounted read-only, credentials are injected only for the
process lifetime, and the instruction and transcript are retained under /logs.
Running inside the task container preserves the declared asset boundary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import IO
from urllib.parse import urlsplit

from container import DEFAULT_RUNTIME, environment_argv, runtime_argv, subprocess_environment
from token_usage import summarize_token_usage, write_token_usage

AGENT_DIR = "/logs/agent"
AGENT_STATE = "state.json"
RETRY_SLEEP_SEC = 60
MAX_RETRY_SLEEP_SEC = 600
# Convenience defaults only. Explicit --model/--reasoning-effort values pass through.
KIMI_DEFAULTS = {
    "api.kimi.com": ("k3", "max"),
    "api.moonshot.cn": ("kimi-k3", "max"),
}


class AgentError(RuntimeError):
    """The agent could not be prepared or run."""


class AgentDeadline(AgentError):
    """The host-enforced exploration deadline expired during an agent exec."""


@dataclass
class AgentApiLease:
    """One cross-process API slot plus the configuration lock that defines its pool."""

    config_handle: IO[str]
    slot_handle: IO[str]
    slot: int
    wait_seconds: float

    def release(self) -> None:
        try:
            self.slot_handle.seek(0)
            self.slot_handle.truncate()
            self.slot_handle.flush()
        finally:
            fcntl.flock(self.slot_handle.fileno(), fcntl.LOCK_UN)
            self.slot_handle.close()
            fcntl.flock(self.config_handle.fileno(), fcntl.LOCK_UN)
            self.config_handle.close()


def validate_reasoning_effort(value: str, agent_name: str = "codex") -> str:
    """Validate that an explicit effort is a usable CLI value."""

    return SPECS[agent_name].validate_reasoning_effort(value)


def validate_model(value: str, agent_name: str = "codex") -> str:
    """Validate that an explicit model is a usable CLI value."""

    return SPECS[agent_name].validate_model(value)


def validate_agent_max_attempts(value: int) -> int:
    """Zero means deadline-bounded retries; positive values cap total attempts."""

    if value < 0:
        raise AgentError("--agent-max-attempts must be zero or a positive integer")
    return value


def validate_agent_api_concurrency(value: int) -> int:
    """Zero disables the independent API semaphore; positive values create slots."""

    if value < 0:
        raise AgentError("--agent-api-concurrency must be zero or a positive integer")
    return value


def validate_agent_api_concurrency_root(value: Path, limit: int) -> Path:
    """An enabled cross-process lock pool must live at an explicit host path."""

    if limit > 0 and not value.is_absolute():
        raise AgentError("--agent-api-concurrency-root must be an absolute path")
    return value


def _lock_with_deadline(handle: IO[str], mode: int, deadline_unix: float | None) -> None:
    while True:
        try:
            fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            if deadline_unix is not None and time.time() >= deadline_unix:
                raise AgentDeadline(
                    "phase deadline reached while waiting for an API slot"
                ) from error
            delay = 0.1
            if deadline_unix is not None:
                delay = min(delay, max(0.0, deadline_unix - time.time()))
            time.sleep(delay)


def _configured_api_limit(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("limit")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, AttributeError) as error:
        raise AgentError(f"invalid API concurrency pool configuration: {path}") from error
    if not isinstance(value, int) or value <= 0:
        raise AgentError(f"invalid API concurrency pool limit in {path}")
    return value


def acquire_agent_api_slot(
    limit: int,
    root: Path,
    *,
    deadline_unix: float | None,
) -> AgentApiLease | None:
    """Acquire one API slot from a host-local or shared-filesystem lock pool.

    The configuration lock is held shared for the lease lifetime, so two active pools
    cannot silently use different limits under the same root. Pointing ``root`` at a
    shared run-control directory extends the same limit across hosts.
    """

    validate_agent_api_concurrency(limit)
    validate_agent_api_concurrency_root(root, limit)
    if limit == 0:
        return None
    started = time.monotonic()
    config_lock: IO[str] | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        config_lock = (root / "config.lock").open("a+", encoding="utf-8")
        config_path = root / "config.json"
        _lock_with_deadline(config_lock, fcntl.LOCK_SH, deadline_unix)
        configured = _configured_api_limit(config_path)
        if configured != limit:
            fcntl.flock(config_lock.fileno(), fcntl.LOCK_UN)
            _lock_with_deadline(config_lock, fcntl.LOCK_EX, deadline_unix)
            configured = _configured_api_limit(config_path)
            if configured is None:
                config_path.write_text(
                    json.dumps({"schema_version": 1, "limit": limit}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif configured != limit:
                raise AgentError(
                    f"API concurrency pool {root} is configured for {configured}, not {limit}"
                )
            # Downgrade while still holding the lock. It remains shared until release,
            # preventing a differently configured process from redefining this pool.
            fcntl.flock(config_lock.fileno(), fcntl.LOCK_SH)

        while True:
            for slot in range(limit):
                slot_handle = (root / f"slot-{slot:04d}.lock").open("a+", encoding="utf-8")
                try:
                    fcntl.flock(
                        slot_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError:
                    slot_handle.close()
                    continue
                slot_handle.seek(0)
                slot_handle.truncate()
                slot_handle.write(
                    json.dumps(
                        {"pid": os.getpid(), "slot": slot, "acquired_unix": time.time()},
                        sort_keys=True,
                    )
                    + "\n"
                )
                slot_handle.flush()
                return AgentApiLease(
                    config_handle=config_lock,
                    slot_handle=slot_handle,
                    slot=slot,
                    wait_seconds=time.monotonic() - started,
                )
            if deadline_unix is not None and time.time() >= deadline_unix:
                raise AgentDeadline("phase deadline reached while waiting for an API slot")
            delay = 0.1
            if deadline_unix is not None:
                delay = min(delay, max(0.0, deadline_unix - time.time()))
            time.sleep(delay)
    except Exception as error:
        if config_lock is not None:
            try:
                fcntl.flock(config_lock.fileno(), fcntl.LOCK_UN)
            finally:
                config_lock.close()
        if isinstance(error, AgentDeadline):
            raise
        raise AgentError(f"agent API concurrency semaphore failed: {error}") from error


@dataclass(frozen=True)
class AgentSpec:
    """One coding agent, and everything the orchestrator needs to know about it."""

    name: str
    # Host path to the binary. Bind-mounted read-only at container_binary.
    binary: Path
    container_binary: str
    # Harbor calls this required_outbound_domains(). Exactly these hostnames and
    # their resolved HTTPS ports are reachable, and nothing else is.
    outbound_domains: tuple[str, ...]
    api_key_env: str
    home_env: str
    default_model: str
    # Model reasoning effort, passed on the command line. "" leaves the CLI default.
    reasoning_effort: str = ""
    # The provider block written into the container's config.toml. base_url is NOT a
    # field: Codex derives it from outbound_domains[0], so the endpoint the CLI is sent
    # to and the host the egress allowlist permits cannot drift apart. A field here
    # would also shadow that property and break the frozen dataclass's __init__.
    provider_name: str = "OpenAI"
    wire_api: str = "responses"
    # Provider provenance is additive; existing Codex/Claude fields remain stable.
    backend: str = "codex"
    protocol: str = "responses"
    context_window_requested: int | None = None
    outbound_ports: tuple[int, ...] = ()

    def outbound_targets(self) -> tuple[str, ...]:
        ports = self.outbound_ports or (443,) * len(self.outbound_domains)
        if len(ports) != len(self.outbound_domains):
            raise AgentError("agent outbound domain/port declarations have different lengths")
        return tuple(
            host if port == 443 else f"{host}:{port}"
            for host, port in zip(self.outbound_domains, ports, strict=True)
        )

    def endpoint_identity(self) -> dict[str, str | int]:
        """Return the credential-free API endpoint identity used for resume checks."""

        parsed = urlsplit(str(getattr(self, "base_url", "")))
        if parsed.scheme != "https" or not parsed.hostname:
            raise AgentError("resolved agent endpoint is not a canonical HTTPS URL")
        try:
            port = parsed.port or 443
        except ValueError as error:
            raise AgentError(f"resolved agent endpoint has an invalid port: {error}") from error
        return {
            "scheme": "https",
            "host": parsed.hostname,
            "port": port,
            "path": parsed.path or "/",
        }

    def validate_model(self, value: str) -> str:
        if not value or not value.strip() or "\0" in value:
            raise AgentError(f"{self.name} model must be a non-empty CLI value")
        return value

    def validate_reasoning_effort(self, value: str) -> str:
        if not value or not value.strip() or "\0" in value:
            raise AgentError(f"{self.name} reasoning effort must be a non-empty CLI value")
        return value

    def api_key(self, environ: dict[str, str] | os._Environ[str] = os.environ) -> str:
        return environ.get(self.api_key_env, "")

    def environment(self, api_key: str, proxy_url: str) -> dict[str, str]:
        return {
            self.api_key_env: api_key,
            # /logs/agent is a distinct host bind for each run. Keeping the CLI home
            # below it makes the session durable for resume without letting two runs
            # share a conversation store.
            self.home_env: f"{AGENT_DIR}/{self.name}-home",
            "HTTPS_PROXY": proxy_url,
            "https_proxy": proxy_url,
            # Keep container-local services on loopback. External destinations
            # still use the explicit proxy and Harbor allowlist above.
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "TMPDIR": "/out/tmp/agent",
        }

    def setup_command(self) -> str:
        raise NotImplementedError

    def run_command(
        self,
        instruction: str,
        model: str,
        output_path: str = f"{AGENT_DIR}/attempt-001.jsonl",
        session_id: str | None = None,
    ) -> str:
        raise NotImplementedError

    def resume_command(
        self, session_id: str, model: str, output_path: str, prompt: str
    ) -> str:
        raise NotImplementedError

    def teardown_command(self) -> str:
        raise NotImplementedError

    def new_session_id(self) -> str | None:
        return None

    def session_id_from_log(self, path: Path) -> str | None:
        return session_id_from_log(path)

    def failure_reason(self, path: Path, status: int) -> str | None:
        return _failure_reason(path, status)


@dataclass(frozen=True)
class Codex(AgentSpec):
    base_url: str = "https://api.openai.com/v1"
    wire_api: str = "responses"

    def setup_command(self) -> str:
        # auth.json under /out rather than /logs/agent, so the secret is not in
        # the same directory as the trajectory that gets kept.
        #
        # CODEX_HOME is isolated, so write the matching provider configuration there.
        # Codex owns the built-in ``openai`` provider name.  Defining a custom
        # block with that name is rejected by recent CLIs before the first API
        # request.  The public OpenAI endpoint therefore selects the built-in
        # provider without overriding it; OpenAI-compatible gateways use a
        # deliberately non-reserved provider ID.
        if self.provider_name == "openai":
            config = (
                'cat > "$CODEX_HOME/config.toml" <<EOF\n'
                'model_provider = "openai"\n'
                "EOF\n"
            )
        else:
            config = (
                'cat > "$CODEX_HOME/config.toml" <<EOF\n'
                f'model_provider = "{self.provider_name}"\n'
                f'[model_providers.{self.provider_name}]\n'
                f'name = "{self.provider_name}"\n'
                f'base_url = "{self.base_url}"\n'
                f'wire_api = "{self.wire_api}"\n'
                "requires_openai_auth = true\n"
                "EOF\n"
            )
        return (
            'mkdir -p "$TMPDIR" /tmp/agent-secrets "$CODEX_HOME"\n'
            'chmod 700 /tmp/agent-secrets\n'
            'cat > /tmp/agent-secrets/auth.json <<EOF\n'
            '{\n'
            f'  "auth_mode": "apikey",\n'
            f'  "{self.api_key_env}": "${{{self.api_key_env}}}"\n'
            '}\n'
            'EOF\n'
            'chmod 600 /tmp/agent-secrets/auth.json\n'
            'ln -sf /tmp/agent-secrets/auth.json "$CODEX_HOME/auth.json"\n'
            # config.toml, so the CLI talks to the gateway the allowlist permits. The
            # block is built above from provider_name/wire_api rather than written with
            # the name inlined here, so one spec can point at a different gateway
            # without a second copy of this heredoc.
            + config
        )

    def _exec_options(self, model: str) -> str:
        return (
            # The container is the sandbox. The agent's own sandbox would be a
            # second one inside it, and it cannot see the GPU or the mounts.
            "--dangerously-bypass-approvals-and-sandbox "
            # /workspace has no .git during the agent phase -- it lives in
            # /opt/harness/git-base until submit.sh puts it back.
            "--skip-git-repo-check "
            f"--model {shlex.quote(model)} "
            "--json "
            "-c web_search=disabled "
            # Passed explicitly rather than left to ~/.codex/config.toml, which is not
            # mounted into the container: the CODEX_HOME the agent runs with is created
            # fresh by setup_command, so anything only in the host config silently does
            # not apply. Reasoning effort changes what the trial measures, so it belongs
            # in the argv the run records.
            + (
                f"-c model_reasoning_effort={shlex.quote(self.reasoning_effort)} "
                if self.reasoning_effort
                else ""
            )
        )

    @staticmethod
    def _record(command: str, output_path: str) -> str:
        # pipefail is load-bearing: without it, tee's zero hides a Codex CLI failure.
        return (
            "set -o pipefail; " + command + " 2>&1 </dev/null | "
            f"stdbuf -oL tee {shlex.quote(output_path)}"
        )

    def run_command(
        self,
        instruction: str,
        model: str,
        output_path: str = f"{AGENT_DIR}/attempt-001.jsonl",
        session_id: str | None = None,
    ) -> str:
        command = (
            f"{self.container_binary} exec "
            + self._exec_options(model)
            + f"-- {shlex.quote(instruction)}"
        )
        return self._record(command, output_path)

    def resume_command(
        self, session_id: str, model: str, output_path: str, prompt: str
    ) -> str:
        # Resume an explicit UUID so concurrent runs cannot select each other's state.
        command = (
            f"{self.container_binary} exec resume "
            + self._exec_options(model)
            + f"{shlex.quote(session_id)} {shlex.quote(prompt)}"
        )
        return self._record(command, output_path)

    def teardown_command(self) -> str:
        # Best effort only, and it must be: this runs through docker exec, and the
        # container may already be gone. Correctness rests on the secret being on a
        # tmpfs, not on this succeeding.
        return 'rm -rf /tmp/agent-secrets "$CODEX_HOME/auth.json"'


def _json_events(path: Path):
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                yield payload


@dataclass(frozen=True)
class Claude(AgentSpec):
    """Native Claude Code with an isolated, resumable configuration directory."""

    base_url: str = "https://api.anthropic.com"

    def api_key(self, environ: dict[str, str] | os._Environ[str] = os.environ) -> str:
        api_key = environ.get("ANTHROPIC_API_KEY", "")
        auth_token = environ.get("ANTHROPIC_AUTH_TOKEN", "")
        if api_key and auth_token and api_key != auth_token:
            raise AgentError(
                "ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are both set but differ"
            )
        return api_key or auth_token

    def environment(self, api_key: str, proxy_url: str) -> dict[str, str]:
        # ANTHROPIC_AUTH_TOKEN is normalized to the documented API-key variable in
        # memory. Neither value is written to Claude's configuration directory.
        environment = super().environment(api_key, proxy_url)
        environment["ANTHROPIC_API_KEY"] = api_key
        environment["ANTHROPIC_BASE_URL"] = self.base_url
        return environment

    def setup_command(self) -> str:
        return 'mkdir -p "$TMPDIR" "$CLAUDE_CONFIG_DIR" && chmod 700 "$CLAUDE_CONFIG_DIR"'

    def _options(self, model: str) -> str:
        return (
            "--bare "
            "--dangerously-skip-permissions "
            "--disable-slash-commands "
            "--no-chrome "
            f"--model {shlex.quote(model)} "
            f"--effort {shlex.quote(self.reasoning_effort)} "
            "--output-format stream-json "
            "--verbose "
        )

    @staticmethod
    def _record(command: str, output_path: str) -> str:
        return (
            "set -o pipefail; " + command + " 2>&1 </dev/null | "
            f"stdbuf -oL tee {shlex.quote(output_path)}"
        )

    def run_command(
        self,
        instruction: str,
        model: str,
        output_path: str = f"{AGENT_DIR}/attempt-001.jsonl",
        session_id: str | None = None,
    ) -> str:
        if not session_id:
            raise AgentError("Claude requires a preallocated session UUID")
        command = (
            f"exec -a {shlex.quote('codex exec claude')} {self.container_binary} "
            + self._options(model)
            + f"--session-id {shlex.quote(session_id)} -p {shlex.quote(instruction)}"
        )
        return self._record(command, output_path)

    def resume_command(
        self, session_id: str, model: str, output_path: str, prompt: str
    ) -> str:
        command = (
            f"exec -a {shlex.quote('codex exec claude')} {self.container_binary} "
            + self._options(model)
            + f"--resume {shlex.quote(session_id)} -p {shlex.quote(prompt)}"
        )
        return self._record(command, output_path)

    def teardown_command(self) -> str:
        return "true"

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    def session_id_from_log(self, path: Path) -> str | None:
        for payload in _json_events(path):
            if payload.get("type") == "system" and payload.get("subtype") == "init":
                value = payload.get("session_id")
                if isinstance(value, str):
                    try:
                        return str(uuid.UUID(value))
                    except ValueError:
                        continue
        return None

    def failure_reason(self, path: Path, status: int) -> str | None:
        result: dict | None = None
        for payload in _json_events(path):
            if payload.get("type") == "result":
                result = payload
        if result is not None:
            if not result.get("is_error") and status == 0:
                return None
            detail = str(result.get("result") or result.get("error") or "Claude result failed")
            api_status = result.get("api_error_status")
            if isinstance(api_status, int):
                detail = f"{detail}; HTTP status {api_status}"
            return detail[:1000]
        if not path.is_file():
            return f"claude exited {status}; attempt log is missing"
        lines = [line.strip() for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines() if line.strip()]
        detail = lines[-1][:400] if lines else "no diagnostic output"
        return f"claude exited {status}: {detail}"

SPECS: dict[str, AgentSpec] = {
    "codex": Codex(
        name="codex",
        # resolve() replaces an npm launcher with its packaged native binary because
        # task images are not required to include Node.js.
        binary=Path.home() / ".local/bin/codex",
        container_binary="/opt/agent/codex",
        # Replaced in resolve() from OPENAI_BASE_URL. Endpoint and allowlist hostname
        # are derived together so a custom provider cannot drift from egress policy.
        outbound_domains=("api.openai.com",),
        api_key_env="OPENAI_API_KEY",
        home_env="CODEX_HOME",
        provider_name="openai",
        default_model="gpt-5.6-sol",
        reasoning_effort="high",
        backend="codex",
        protocol="responses",
    ),
    "claude": Claude(
        name="claude",
        binary=Path.home() / ".local/bin/claude",
        container_binary="/opt/agent/claude",
        # Replaced in resolve() from ANTHROPIC_BASE_URL. Keeping the endpoint and
        # allowlist derivation in one function prevents gateway/egress drift.
        outbound_domains=("api.anthropic.com",),
        api_key_env="ANTHROPIC_API_KEY",
        home_env="CLAUDE_CONFIG_DIR",
        default_model="claude-opus-5",
        reasoning_effort="high",
        provider_name="anthropic",
        wire_api="anthropic",
        backend="anthropic",
        protocol="anthropic",
    ),
}


def _https_endpoint(value: str, variable: str) -> tuple[str, str, int]:
    endpoint = value.rstrip("/")
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as error:
        raise AgentError(
            f"{variable} must be an HTTPS URL with a valid port (443 by default), "
            "without credentials, query parameters, or a fragment"
        ) from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and port <= 0)
    ):
        raise AgentError(
            f"{variable} must be an HTTPS URL with a valid port (443 by default), "
            "without credentials, query parameters, or a fragment"
        )
    return endpoint, parsed.hostname, port or 443


def _claude_endpoint(value: str) -> tuple[str, str, int]:
    return _https_endpoint(value, "ANTHROPIC_BASE_URL")


def _codex_endpoint(value: str) -> tuple[str, str, int]:
    return _https_endpoint(value, "OPENAI_BASE_URL")


def _native_binary(launcher: Path) -> Path | None:
    """Find the platform binary behind an npm launcher script.

    npm launchers require Node.js, which task images need not provide. Resolve a
    self-contained platform binary relative to the launcher instead of depending on a
    host-specific installation path.
    """

    if launcher.suffix != ".js":
        return None
    # .../node_modules/@openai/codex/bin/codex.js -> .../node_modules/@openai
    scope = launcher.parent.parent.parent
    for triple in ("x86_64-unknown-linux-musl", "x86_64-unknown-linux-gnu"):
        candidate = scope / "codex-linux-x64" / "vendor" / triple / "bin" / "codex"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_metadata(name: str) -> AgentSpec:
    """Resolve provider defaults and endpoint metadata without requiring a CLI binary."""

    if name not in SPECS:
        raise AgentError(f"unknown agent {name!r}; known: {sorted(SPECS)}")
    spec = SPECS[name]
    if isinstance(spec, Codex):
        configured = os.environ.get("OPENAI_BASE_URL")
        legacy_domain = os.environ.get("AGENT_OUTBOUND_DOMAIN")
        if not configured and legacy_domain:
            configured = f"https://{legacy_domain}/v1"
        base_url, outbound_domain, outbound_port = _codex_endpoint(configured or spec.base_url)
        provider_name = (
            "openai"
            if base_url == "https://api.openai.com/v1"
            else "ai4ai_openai_compatible"
        )
        spec = replace(
            spec,
            base_url=base_url,
            outbound_domains=(outbound_domain,),
            outbound_ports=(outbound_port,),
            provider_name=provider_name,
        )
    elif isinstance(spec, Claude):
        base_url, outbound_domain, outbound_port = _claude_endpoint(
            os.environ.get("ANTHROPIC_BASE_URL", spec.base_url)
        )
        spec = replace(
            spec,
            base_url=base_url,
            outbound_domains=(outbound_domain,),
            outbound_ports=(outbound_port,),
        )
        if outbound_domain in KIMI_DEFAULTS:
            default_model, default_effort = KIMI_DEFAULTS[outbound_domain]
            spec = replace(
                spec,
                backend="kimi",
                provider_name="kimi",
                protocol="anthropic",
                default_model=default_model,
                reasoning_effort=default_effort,
            )
    return spec


def resolve(name: str) -> AgentSpec:
    """Resolve an agent for execution, including its runnable host binary."""

    spec = resolve_metadata(name)
    variable = f"{spec.name.upper()}_BINARY"
    configured = os.environ.get(variable)
    if configured is not None:
        if not configured.strip() or "\0" in configured:
            raise AgentError(f"{variable} must name a non-empty executable path")
        candidate = Path(configured).expanduser()
    else:
        on_path = shutil.which(spec.name)
        candidate = Path(on_path) if on_path else spec.binary
    binary = candidate.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise AgentError(
            f"{spec.name} binary is not executable: {binary}. Set {variable}, add "
            f"{spec.name} to PATH, or install it at {spec.binary}."
        )
    spec = replace(spec, binary=binary)
    native = _native_binary(binary)
    if native is not None:
        spec = replace(spec, binary=native)
    elif binary.suffix == ".js":
        raise AgentError(
            f"{binary} is a node launcher and no platform binary was found beside it. "
            "The task image has no node, so the launcher cannot run inside it. Install "
            f"the {spec.name} native package, or point {spec.name.upper()}_BINARY at a "
            "self-contained executable."
        )
    return spec


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exec(
    container: str,
    command: str,
    environment: dict[str, str],
    *,
    runtime: str = DEFAULT_RUNTIME,
    workdir: str = "/workspace",
    check: bool = True,
    stream: bool = False,
    timeout_seconds: float | None = None,
) -> int:
    argv = [*runtime_argv(runtime), "exec", "--workdir", workdir]
    process_environment = subprocess_environment(environment)
    argv.extend(environment_argv(environment))
    argv.extend((container, "bash", "-lc", command))
    try:
        if stream:
            result = subprocess.run(
                argv,
                check=False,
                env=process_environment,
                timeout=timeout_seconds,
            )
        else:
            result = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                env=process_environment,
                timeout=timeout_seconds,
            )
            if result.returncode != 0 and check:
                raise AgentError(
                    f"exec failed ({result.returncode}): {result.stderr.strip()[:400]}"
                )
    except subprocess.TimeoutExpired as error:
        raise AgentDeadline("agent exec reached the exploration deadline") from error
    if result.returncode != 0 and check:
        raise AgentError(f"exec failed with {result.returncode}")
    return result.returncode


def verify_binary(spec: AgentSpec, container: str, *, runtime: str = DEFAULT_RUNTIME) -> str:
    """Harbor's binary_check_command: is the agent actually there and runnable?

    Worth doing before anything else -- a bind mount that silently did not land
    otherwise surfaces as a confusing failure several steps later.
    """

    result = subprocess.run(
        [*runtime_argv(runtime), "exec", container, spec.container_binary, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AgentError(
            f"{spec.name} is not runnable at {spec.container_binary} inside {container}: "
            f"{(result.stderr or result.stdout).strip()[:300]}"
        )
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "(no version)"


def session_id_from_log(path: Path) -> str | None:
    """Extract Codex's durable conversation id from a thread.started JSON event."""

    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("type") != "thread.started":
            continue
        value = payload.get("thread_id") or payload.get("session_id")
        if not isinstance(value, str):
            continue
        try:
            return str(uuid.UUID(value))
        except ValueError:
            continue
    return None


def _event_in_log(path: Path, event_type: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if json.loads(line).get("type") == event_type:
                return True
        except (ValueError, AttributeError):
            continue
    return False


def _turn_failure_from_log(path: Path) -> str | None:
    if not path.is_file():
        return None
    failure: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("type") == "turn.failed":
            error = payload.get("error", {})
            if not isinstance(error, dict):
                failure = str(error or line)
                continue
            parts = [str(error.get("message") or line)]
            status = error.get("status_code", error.get("status"))
            if isinstance(status, int) or (
                isinstance(status, str) and status.isdigit()
            ):
                parts.append(f"HTTP status {status}")
            headers = error.get("headers")
            if isinstance(headers, dict):
                retry_after = next(
                    (
                        value
                        for key, value in headers.items()
                        if str(key).lower() == "retry-after"
                    ),
                    None,
                )
                if retry_after is not None:
                    parts.append(f"Retry-After: {retry_after}")
            failure = "; ".join(parts)
    return failure


def _failure_reason(path: Path, status: int) -> str | None:
    failure = _turn_failure_from_log(path)
    if failure:
        return failure
    if status == 0 and _event_in_log(path, "turn.completed"):
        return None
    if not path.is_file():
        return f"codex exited {status}; attempt log is missing"
    lines = [line.strip() for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines() if line.strip()]
    detail = lines[-1][:400] if lines else "no diagnostic output"
    return f"codex exited {status}: {detail}"


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = summarize_token_usage(
        path.parent,
        session_id=str(state.get("session_id")) if state.get("session_id") else None,
        agent=str(state.get("agent") or "codex"),
    )
    state["token_usage"] = usage
    write_token_usage(path.parent / "token_usage.json", usage)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _persist_started_session(
    attempt_path: Path,
    state_path: Path,
    state: dict,
    stopped: threading.Event,
    session_parser=session_id_from_log,
) -> None:
    """Persist thread.started while Codex is still running.

    ``docker exec`` streams the JSONL to a host bind but blocks until Codex exits. A
    node or container loss in that interval must not also lose the only durable session
    identifier, so a small host-side watcher records it as soon as the event lands.
    """

    while not stopped.is_set():
        session_id = session_parser(attempt_path)
        if session_id:
            state["session_id"] = session_id
            state["session_initialized"] = True
            current = _load_state(state_path) or dict(state)
            current["session_id"] = session_id
            current["session_initialized"] = True
            _write_state(state_path, current)
            return
        stopped.wait(0.1)


def _capture_exec(
    container: str, command: str, *, runtime: str = DEFAULT_RUNTIME, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*runtime_argv(runtime), "exec", container, "bash", "-lc", command],
        check=False, capture_output=True, text=True, timeout=timeout,
    )


def _workspace_fingerprint(container: str, *, runtime: str = DEFAULT_RUNTIME) -> str | None:
    script = "python3 /opt/harness/workspace_fingerprint.py /workspace"
    try:
        result = _capture_exec(container, script, runtime=runtime, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return value if result.returncode == 0 and len(value) == 64 else None


def _workspace_marker(container: str, run_id: str, *, runtime: str) -> bool:
    result = _capture_exec(
        container, "cat /tmp/ai4ai-agent-run-id 2>/dev/null", runtime=runtime
    )
    return result.returncode == 0 and result.stdout.strip() == run_id


def _set_workspace_marker(container: str, run_id: str, *, runtime: str) -> bool:
    result = _capture_exec(
        container,
        f"printf %s {shlex.quote(run_id)} > /tmp/ai4ai-agent-run-id",
        runtime=runtime,
    )
    return result.returncode == 0


def _checkpoint_workspace(
    container: str, attempt: int, *, runtime: str
) -> tuple[str, str] | None:
    container_path = f"{AGENT_DIR}/checkpoints/attempt-{attempt:03d}.patch"
    script = (
        f"mkdir -p {AGENT_DIR}/checkpoints\n"
        "find /workspace -type d -name __pycache__ -prune -exec rm -rf {} + "
        "2>/dev/null || true\n"
        "find /workspace -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -delete "
        "2>/dev/null || true\n"
        "rm -rf /tmp/ai4ai-workspace-git\n"
        "cp -a /opt/harness/git-base/.git /tmp/ai4ai-workspace-git\n"
        "chmod -R u+rwX /tmp/ai4ai-workspace-git\n"
        "GIT_DIR=/tmp/ai4ai-workspace-git GIT_WORK_TREE=/workspace git add -A -f\n"
        "GIT_DIR=/tmp/ai4ai-workspace-git GIT_WORK_TREE=/workspace "
        "git diff --cached --binary -- . "
        "':(exclude,glob)**/__pycache__/**' "
        "':(exclude,glob)**/*.pyc' ':(exclude,glob)**/*.pyo' "
        "':(exclude,glob)*.pyc' ':(exclude,glob)*.pyo' "
        f"> {container_path}\n"
        f"sha256sum {container_path} | awk '{{print $1}}'\n"
    )
    try:
        result = _capture_exec(container, script, runtime=runtime, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    digest = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or len(digest) != 64:
        return None
    return container_path, digest


def _restore_workspace_checkpoint(
    container: str, checkpoint: str, *, runtime: str
) -> bool:
    script = (
        f"test -f {shlex.quote(checkpoint)} || exit 3\n"
        f"if [ -s {shlex.quote(checkpoint)} ]; then "
        "git -C /workspace apply --binary --exclude='**/__pycache__/**' "
        "--exclude='*.pyc' --exclude='*.pyo' "
        f"{shlex.quote(checkpoint)}; fi\n"
    )
    try:
        result = _capture_exec(container, script, runtime=runtime, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _resume_failure(failure: str | None, status: int, log: Path) -> bool:
    lowered = (failure or "").lower()
    if any(marker in lowered for marker in (
        "session not found", "thread not found", "no rollout found", "unknown session",
    )):
        return True
    # A transport failure can end the CLI before it emits a second turn.started event.
    # That is still resumable; only an explicit session lookup failure is terminal here.
    return False


_HTTP_STATUS = re.compile(
    r"(?i)(?:http(?:/\d(?:\.\d)?)?\s+|http\s+status(?:\s+code)?\s*[:=]?\s*|"
    r"response\s+status(?:\s+code)?\s*[:=]?\s*|status(?:\s+code)?\s*[:=]?\s*)"
    r"(?P<code>\d{3})\b"
)
_HTTP_STATUS_WITH_REASON = re.compile(
    r"(?i)(?:^|\s)(?P<code>429|502|503|504)\s+"
    r"(?:too\s+many\s+requests|bad\s+gateway|service\s+unavailable|gateway\s+timeout)\b"
)
_RETRY_AFTER_NUMBER = re.compile(
    r"(?i)\bretry[-_ ]after\b[\"']?\s*(?:[:=]\s*)?[\"']?"
    r"(?P<seconds>\d+(?:\.\d+)?)"
)
_RETRY_AFTER_VALUE = re.compile(r"(?i)\bretry-after\b\s*:\s*(?P<value>[^\r\n]+)")


def _http_status_codes(failure: str) -> set[int]:
    """Return explicit HTTP status codes, without treating arbitrary numbers as codes."""

    return {
        int(match.group("code"))
        for pattern in (_HTTP_STATUS, _HTTP_STATUS_WITH_REASON)
        for match in pattern.finditer(failure)
    }


def _retry_after_seconds(failure: str, *, now_unix: float | None = None) -> int | None:
    """Parse a numeric or HTTP-date Retry-After value from a failure message."""

    match = _RETRY_AFTER_NUMBER.search(failure)
    if match:
        return max(0, math.ceil(float(match.group("seconds"))))
    header = _RETRY_AFTER_VALUE.search(failure)
    if not header:
        return None
    raw = header.group("value").strip().strip("\"'")
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return max(0, math.ceil(parsed.timestamp() - (now_unix or time.time())))


def _retry_backoff_seconds(failure: str, transient_index: int) -> tuple[int, int | None]:
    retry_after = _retry_after_seconds(failure)
    exponential = min(
        MAX_RETRY_SLEEP_SEC,
        RETRY_SLEEP_SEC * (2 ** max(0, transient_index - 1)),
    )
    return (retry_after if retry_after is not None else exponential), retry_after


def run_agent(
    spec: AgentSpec,
    container: str,
    *,
    instruction: str,
    model: str,
    proxy_url: str,
    api_key: str,
    agent_log_dir: Path,
    runtime: str = DEFAULT_RUNTIME,
    deadline_unix: float | None = None,
    max_attempts: int = 0,
    api_concurrency: int = 0,
    api_concurrency_root: Path = Path("/tmp/ai4ai-agent-api"),
) -> int:
    """Run Codex and resume only its original session after transient failures.

    Workspace and conversation continuity are separate checks. A persisted Codex UUID
    is not enough: before every resume the exact /workspace tree must match the patch
    checkpoint recorded after the previous attempt. A restarted container restores that
    checkpoint first; an unprovable restoration is `resume_failed`, never a new session.
    """

    validate_agent_max_attempts(max_attempts)
    validate_agent_api_concurrency(api_concurrency)
    validate_agent_api_concurrency_root(api_concurrency_root, api_concurrency)
    if max_attempts == 0 and deadline_unix is None:
        raise AgentError(
            "--agent-max-attempts 0 requires the original absolute phase deadline"
        )
    environment = spec.environment(api_key, proxy_url)
    version = verify_binary(spec, container, runtime=runtime)
    print(f"agent: {spec.name} {version} in {container}")
    print(f"agent: model={model} reasoning_effort={spec.reasoning_effort or 'default'}")
    print(
        "agent: retry policy "
        f"max_attempts={max_attempts or 'deadline'} "
        f"api_concurrency={api_concurrency or 'unlimited'} "
        f"api_concurrency_root={api_concurrency_root}"
    )

    agent_log_dir.mkdir(parents=True, exist_ok=True)
    state_path = agent_log_dir / AGENT_STATE
    previous = _load_state(state_path)
    if previous and previous.get("status") == "resume_failed":
        print("agent: previous state is resume_failed; refusing to open a new session")
        return 1
    requested_policy = {
        "max_attempts": max_attempts,
        "api_concurrency": api_concurrency,
        "api_concurrency_root": str(api_concurrency_root),
        "deadline_unix": deadline_unix,
    }
    session_identity = {
        "agent": spec.name,
        "endpoint": spec.endpoint_identity(),
        "backend": spec.backend,
        "provider": spec.provider_name,
        "protocol": spec.protocol,
        "wire_api": spec.wire_api,
        "binary_sha256": _binary_sha256(spec.binary),
        "agent_version": version,
    }
    if previous:
        prior_policy = previous.get("retry_policy")
        if prior_policy is not None and prior_policy != requested_policy:
            previous.update(
                status="resume_failed",
                last_failure_reason="retry policy changed while resuming the same run",
            )
            _write_state(state_path, previous)
            print("agent: resume_failed -- retry policy changed")
            return 1
        if previous.get("attempts") or previous.get("session_initialized"):
            prior_identity = previous.get("session_identity")
            if prior_identity != session_identity:
                previous.update(
                    status="resume_failed",
                    last_failure_reason=(
                        "agent implementation or endpoint changed while resuming the same session"
                    ),
                )
                _write_state(state_path, previous)
                print("agent: resume_failed -- agent session identity changed")
                return 1

    _exec(
        container,
        f"mkdir -p {AGENT_DIR} {shlex.quote(environment[spec.home_env])}",
        environment,
        runtime=runtime,
    )
    # Write the instruction the agent will be given, before giving it, so the
    # record exists even if the run dies immediately.
    _exec(
        container,
        f"cat > {AGENT_DIR}/instruction.md <<'AI4AI_INSTRUCTION_EOF'\n"
        f"{instruction}\nAI4AI_INSTRUCTION_EOF",
        environment,
        runtime=runtime,
    )
    _exec(container, spec.setup_command(), environment, runtime=runtime)

    run_id = (
        str(previous.get("run_id"))
        if previous and previous.get("run_id")
        else str(uuid.uuid4())
    )
    preallocated_session = spec.new_session_id() if not previous else None
    state = previous or {
        "schema_version": 1,
        "run_id": run_id,
        "session_id": preallocated_session,
        "session_initialized": False,
        "resume_count": 0,
        "attempts": [],
        "last_failure_reason": None,
        "status": "running",
        "retry_policy": requested_policy,
    }
    state["retry_policy"] = requested_policy
    state["session_identity"] = session_identity
    state["agent"] = spec.name
    state["agent_version"] = version
    state["codex_version"] = version if spec.name == "codex" else None
    state["backend"] = spec.backend
    state["provider"] = spec.provider_name
    state["protocol"] = spec.protocol
    state["endpoint_host"] = spec.outbound_domains[0] if spec.outbound_domains else None
    state["endpoint"] = session_identity["endpoint"]
    state["endpoint_port"] = session_identity["endpoint"]["port"]
    state["endpoint_path"] = session_identity["endpoint"]["path"]
    state["wire_api"] = spec.wire_api
    state["binary_sha256"] = session_identity["binary_sha256"]
    state["context_window_requested"] = spec.context_window_requested
    state["requested_model"] = model
    state["requested_effort"] = spec.reasoning_effort
    state.setdefault("session_initialized", bool(state.get("session_id")))
    state["status"] = "running"
    _write_state(state_path, state)

    session_id = state.get("session_id")
    attempts: list[dict] = list(state.get("attempts") or [])
    next_attempt = len(attempts) + 1
    # Claude owns its UUID before the first request. If a transient attempt dies
    # before `system/init`, the UUID is still the only session it may try again;
    # persisted attempts also require the same workspace checkpoint restoration.
    resume = bool(
        session_id and (state.get("session_initialized") or state.get("attempts"))
    )
    expected_workspace: str | None = None

    if resume:
        latest = attempts[-1] if attempts else {}
        checkpoint = latest.get("workspace_checkpoint")
        expected_workspace = latest.get("workspace_sha256")
        if not checkpoint or not expected_workspace:
            state.update(
                status="resume_failed",
                last_failure_reason="persisted session has no workspace checkpoint",
            )
            _write_state(state_path, state)
            print("agent: resume_failed -- persisted session has no workspace checkpoint")
            return 1
        if not _restore_workspace_checkpoint(container, str(checkpoint), runtime=runtime):
            state.update(
                status="resume_failed",
                last_failure_reason="workspace checkpoint could not be restored",
            )
            _write_state(state_path, state)
            print("agent: resume_failed -- workspace checkpoint could not be restored")
            return 1
        restored = _workspace_fingerprint(container, runtime=runtime)
        if restored != expected_workspace:
            state.update(
                status="resume_failed",
                last_failure_reason=(
                    f"workspace checkpoint mismatch: expected {expected_workspace}, got {restored}"
                ),
            )
            _write_state(state_path, state)
            print(f"agent: resume_failed -- {state['last_failure_reason']}")
            return 1

        retry_not_before = latest.get("retry_not_before_unix")
        if isinstance(retry_not_before, (int, float)):
            delay = max(0.0, float(retry_not_before) - time.time())
            if deadline_unix is not None and time.time() + delay >= deadline_unix:
                state.update(
                    status="timed_out",
                    last_failure_reason="phase deadline reached during persisted retry backoff",
                )
                _write_state(state_path, state)
                return 124
            if delay:
                print(f"agent: preserving persisted Retry-After/backoff for {delay:.1f}s")
                time.sleep(delay)

    if not _set_workspace_marker(container, run_id, runtime=runtime):
        state.update(status="resume_failed", last_failure_reason="could not establish run marker")
        _write_state(state_path, state)
        print("agent: resume_failed -- could not establish run marker")
        return 1

    started = time.monotonic()
    status = 1
    failure: str | None = None
    timed_out = False
    attempt = next_attempt
    if max_attempts > 0 and attempt > max_attempts:
        failure = f"agent attempt limit {max_attempts} was already exhausted"
        state.update(status="failed", last_failure_reason=failure)
        _write_state(state_path, state)
        print(f"agent: FAILED -- {failure}")
        return 1
    try:
        while max_attempts == 0 or attempt <= max_attempts:
            remaining = deadline_unix - time.time() if deadline_unix else None
            if remaining is not None and remaining <= 0:
                timed_out = True
                failure = "phase deadline reached before the next agent attempt"
                state.update(status="timed_out", last_failure_reason=failure)
                _write_state(state_path, state)
                break
            attempt_path = agent_log_dir / f"attempt-{attempt:03d}.jsonl"
            container_attempt_path = f"{AGENT_DIR}/attempt-{attempt:03d}.jsonl"
            if attempt_path.exists():
                state.update(
                    status="resume_failed",
                    last_failure_reason=f"attempt log already exists: {attempt_path.name}",
                )
                _write_state(state_path, state)
                print(f"agent: resume_failed -- {state['last_failure_reason']}")
                return 1

            if resume:
                current = _workspace_fingerprint(container, runtime=runtime)
                if (
                    not session_id
                    or not _workspace_marker(container, run_id, runtime=runtime)
                    or current != expected_workspace
                ):
                    state.update(
                        status="resume_failed",
                        last_failure_reason=(
                            "workspace continuity check failed before resume: "
                            f"expected {expected_workspace}, got {current}"
                        ),
                    )
                    _write_state(state_path, state)
                    print(f"agent: resume_failed -- {state['last_failure_reason']}")
                    return 1
                state["resume_count"] = int(state.get("resume_count", 0)) + 1
                command = spec.resume_command(
                    str(session_id),
                    model,
                    container_attempt_path,
                    "Continue the same task from the preserved workspace state.",
                )
                print(f"agent: resuming session {session_id} (attempt {attempt})")
            else:
                command = spec.run_command(
                    instruction,
                    model,
                    container_attempt_path,
                    session_id=str(session_id) if session_id else None,
                )

            attempt_started = time.time()
            api_slot: int | None = None
            api_wait_started = time.monotonic()
            api_wait_seconds = 0.0
            lease: AgentApiLease | None = None
            watcher_stop = threading.Event()
            watcher = None
            if not resume:
                watcher = threading.Thread(
                    target=_persist_started_session,
                    args=(
                        attempt_path,
                        state_path,
                        state,
                        watcher_stop,
                        spec.session_id_from_log,
                    ),
                    name=f"{spec.name}-session-watcher",
                    daemon=True,
                )
                watcher.start()
            try:
                try:
                    lease = acquire_agent_api_slot(
                        api_concurrency,
                        api_concurrency_root,
                        deadline_unix=deadline_unix,
                    )
                    if lease is not None:
                        api_slot = lease.slot
                        api_wait_seconds = lease.wait_seconds
                    else:
                        api_wait_seconds = time.monotonic() - api_wait_started
                    exec_remaining = (
                        deadline_unix - time.time() if deadline_unix is not None else None
                    )
                    if exec_remaining is not None and exec_remaining <= 0:
                        raise AgentDeadline(
                            "phase deadline reached while waiting for an API slot"
                        )
                    status = _exec(
                        container,
                        command,
                        environment,
                        runtime=runtime,
                        check=False,
                        stream=True,
                        timeout_seconds=exec_remaining,
                    )
                    failure = spec.failure_reason(attempt_path, status)
                except AgentDeadline as error:
                    status = 124
                    timed_out = True
                    api_wait_seconds = time.monotonic() - api_wait_started
                    failure = str(error) or (
                        "phase deadline reached while waiting for or running the agent"
                    )
                except AgentError as error:
                    status = 1
                    api_wait_seconds = time.monotonic() - api_wait_started
                    failure = str(error)
            except (OSError, subprocess.SubprocessError) as error:
                status = 1
                failure = f"container lost before attempt completed: {error}"
            finally:
                if lease is not None:
                    lease.release()
                watcher_stop.set()
                if watcher is not None:
                    watcher.join(timeout=2)
            logged_session = spec.session_id_from_log(attempt_path)
            if not resume:
                if logged_session and session_id and logged_session != session_id:
                    failure = (
                        f"new session returned a different session: expected {session_id}, "
                        f"got {logged_session}"
                    )
                session_id = logged_session or session_id or state.get("session_id")
                state["session_id"] = session_id
                state["session_initialized"] = bool(logged_session)
            elif logged_session and logged_session != session_id:
                failure = (
                    f"resume returned a different session: expected {session_id}, "
                    f"got {logged_session}"
                )

            attempt_ended = time.time()
            workspace_at_end: str | None = None
            if not timed_out and "container lost" not in (failure or ""):
                workspace_at_end = _workspace_fingerprint(container, runtime=runtime)
            entry: dict = {
                "attempt": attempt,
                "log": attempt_path.name,
                "resumed": resume,
                "started_unix": attempt_started,
                "ended_unix": attempt_ended,
                "remaining_seconds_at_start": (
                    max(0.0, deadline_unix - attempt_started)
                    if deadline_unix is not None
                    else None
                ),
                "remaining_seconds_at_end": (
                    max(0.0, deadline_unix - attempt_ended)
                    if deadline_unix is not None
                    else None
                ),
                "exit_status": status,
                "failure_reason": failure,
                "session_id": session_id,
                "workspace_sha256": workspace_at_end,
                "workspace_checkpoint": None,
                "workspace_checkpoint_sha256": None,
                "retry_after_seconds": None,
                "backoff_seconds": None,
                "retry_not_before_unix": None,
                "api_concurrency_limit": api_concurrency,
                "api_concurrency_root": str(api_concurrency_root),
                "api_slot": api_slot,
                "api_slot_wait_seconds": round(api_wait_seconds, 6),
            }

            if timed_out:
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="timed_out",
                    last_failure_reason=failure,
                    session_id=session_id,
                )
                _write_state(state_path, state)
                break

            if not failure:
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="completed",
                    last_failure_reason=None,
                    session_id=session_id,
                )
                _write_state(state_path, state)
                break

            if "container lost" in (failure or "") or (
                resume and _resume_failure(failure, status, attempt_path)
            ):
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="resume_failed",
                    last_failure_reason=failure,
                    session_id=session_id,
                )
                _write_state(state_path, state)
                print(f"agent: resume_failed -- {failure[:200]}")
                return 1

            transient = _is_transient(failure)
            entry["transient"] = transient
            if not transient or (max_attempts > 0 and attempt >= max_attempts):
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="failed",
                    last_failure_reason=failure,
                    session_id=session_id,
                )
                _write_state(state_path, state)
                break

            if not session_id:
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="resume_failed",
                    last_failure_reason=(
                        f"transient failure had no thread.started session id: {failure}"
                    ),
                )
                _write_state(state_path, state)
                print(f"agent: resume_failed -- {state['last_failure_reason'][:200]}")
                return 1

            checkpoint = _checkpoint_workspace(container, attempt, runtime=runtime)
            workspace = workspace_at_end
            if not checkpoint or not workspace:
                attempts.append(entry)
                state.update(
                    attempts=attempts,
                    status="resume_failed",
                    last_failure_reason="could not checkpoint workspace before resume",
                    session_id=session_id,
                )
                _write_state(state_path, state)
                print("agent: resume_failed -- could not checkpoint workspace before resume")
                return 1

            transient_index = sum(
                1 for recorded in attempts if recorded.get("transient")
            ) + 1
            backoff_seconds, retry_after_seconds = _retry_backoff_seconds(
                failure, transient_index
            )
            entry["retry_after_seconds"] = retry_after_seconds
            entry["backoff_seconds"] = backoff_seconds
            entry["retry_not_before_unix"] = time.time() + backoff_seconds
            entry["workspace_checkpoint"] = checkpoint[0]
            entry["workspace_checkpoint_sha256"] = checkpoint[1]
            entry["workspace_sha256"] = workspace
            attempts.append(entry)
            expected_workspace = workspace
            state.update(
                attempts=attempts,
                status="running",
                last_failure_reason=failure,
                session_id=session_id,
            )
            _write_state(state_path, state)
            # The original phase deadline remains authoritative across resumes. Release
            # the API slot before sleeping so backoff does not consume shared capacity.
            minutes = (time.monotonic() - started) / 60
            print(
                f"agent: transient failure at {minutes:.1f} min "
                f"(attempt {attempt}/{max_attempts or 'deadline'}) -- {failure[:120]}"
            )
            print(
                f"agent: resuming {session_id} in {backoff_seconds}s; "
                f"workspace {workspace[:12]} checkpointed"
            )
            if deadline_unix:
                remaining = deadline_unix - time.time()
                if remaining <= 1:
                    timed_out = True
                    failure = "phase deadline reached before transient resume"
                    state.update(status="timed_out", last_failure_reason=failure)
                    _write_state(state_path, state)
                    break
                time.sleep(min(backoff_seconds, max(0, remaining)))
            else:
                time.sleep(backoff_seconds)
            resume = True
            attempt += 1
    finally:
        # Runs on a crash and on Ctrl-C. The key must not outlive the run.
        _exec(
            container,
            spec.teardown_command(),
            {spec.home_env: AGENT_DIR},
            runtime=runtime,
            check=False,
        )

    minutes = (time.monotonic() - started) / 60
    if timed_out:
        print(f"agent: TIMED OUT after {minutes:.1f} min")
        return 124
    if failure:
        # codex exec returns 0 even when the turn failed -- an unusable API key
        # exits 0 with `turn.failed` in the stream. Reporting that as success
        # would make a run that never started look like a run that found nothing.
        print(f"agent: FAILED after {minutes:.1f} min -- {failure[:200]}")
        return 1 if status == 0 else status
    print(f"agent: exit {status} after {minutes:.1f} min")
    return status


def _is_transient(failure: str) -> bool:
    """Whether a turn failure is worth another attempt.

    Deliberately a small list of upstream conditions rather than "anything that is not
    obviously fatal". An unusable key or unknown model remains terminal regardless of
    the configured attempt policy.
    """

    lowered = failure.lower()
    # Never spend the retry budget on an answer that proves the request itself is
    # invalid. In particular, a config parser can mention a timeout while still being
    # a deterministic local error, so fatal classes are checked before transport words.
    status_codes = _http_status_codes(failure)
    if status_codes & {400, 401, 403, 404}:
        return False
    if any(
        marker in lowered
        for marker in (
            "invalid api key", "authentication", "unauthorized", "not authorized", "forbidden",
            "unknown model", "model does not exist", "model not found",
            "unknown argument", "configuration error", "configuration failed",
            "config error", "invalid configuration", "config.toml", "toml parse",
            "failed to parse config", "unrecognized option",
        )
    ):
        return False
    if 429 in status_codes or any(500 <= code < 600 for code in status_codes):
        return True
    return any(
        marker in lowered
        for marker in (
            "at capacity",
            "rate limit",
            "server_error",
            "internal server error",
            "timeout",
            "temporarily unavailable",
            "overloaded",
            # Transport failures have no authoritative API response and are safe to retry
            # within the original phase deadline.
            "stream disconnected",
            "error sending request",
            "connection closed",
            "connection reset",
            "could not resolve host",
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided",
        )
    )


def _turn_failure(container: str, *, runtime: str = "docker") -> str | None:
    """Look for a `turn.failed` event in the agent's own output stream.

    Container shutdown can race this diagnostic exec. A runtime error is not evidence that
    the agent itself failed, so an unreadable stream returns no agent failure.
    """

    result = subprocess.run(
        [
            # A runtime may be a wrapper plus arguments, not a single executable path.
            *runtime_argv(runtime),
            "exec",
            container,
            "bash",
            "-lc",
            f"grep -h '\"turn.failed\"' {AGENT_DIR}/attempt-*.jsonl 2>/dev/null | tail -1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        # grep exits 1 on no match, which is a real "no failure found". Anything else is
        # the exec itself failing -- gone container, dead daemon -- so there is nothing to
        # report either way.
        combined = (result.stderr or "") + (result.stdout or "")
        if "exec failed" in combined or "No such container" in combined or "setns" in combined:
            return None
    line = result.stdout.strip()
    if not line:
        return None
    try:
        import json

        payload = json.loads(line)
        return str(payload.get("error", {}).get("message", line))
    except (ValueError, AttributeError):
        return line
