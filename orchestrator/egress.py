"""Allow an isolated task container to reach only its configured model API.

The task network remains internal and DNS-free. A host-side CONNECT proxy resolves
and permits the exact API hostname while rejecting every other destination.
"""

from __future__ import annotations

import os
import selectors
import socket
import subprocess
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from container import DEFAULT_RUNTIME, runtime_argv

CONNECT_TIMEOUT = 20.0
# Long reasoning responses may be silent for minutes. Application-level inactivity is
# therefore not a dead-peer signal; TCP keepalive handles that distinction below.
IDLE_TIMEOUT = 14400.0
# Detect a genuinely dead peer without relying on silence. Values in seconds: start probing
# after 60 s idle, probe every 15 s, give up after 8 failures -- so a dead peer is noticed
# in about three minutes while a thinking model is left alone.
KEEPALIVE_IDLE_SEC = 60
KEEPALIVE_INTERVAL_SEC = 15
KEEPALIVE_FAILURES = 8
RELAY_CHUNK = 65536
# A CONNECT line is short. Anything longer is not a client we want to talk to.
MAX_REQUEST_BYTES = 8192

# Optional upstream CONNECT proxy. Direct connection is the portable default; deployments
# without outbound routing can set one of the supported proxy environment variables.
UPSTREAM_PROXY = os.environ.get("AI4AI_EGRESS_UPSTREAM", "")


class EgressError(RuntimeError):
    """The egress network or proxy could not be set up."""


@dataclass
class EgressProxy:
    """An HTTPS CONNECT proxy that allows an exact set of hostname/port targets.

    Plain HTTP proxying is not implemented. The only client is an agent CLI
    talking to an HTTPS API, and refusing everything else keeps the reachable
    surface to one verb. A bare hostname means its default HTTPS port 443;
    non-default ports must be present explicitly as ``hostname:port``.
    """

    allowed_hosts: frozenset[str]
    bind_host: str
    port: int = 0
    log_path: Path | None = None
    # host:port of a CONNECT proxy to reach allowed hosts through; "" dials direct.
    upstream: str = UPSTREAM_PROXY
    _server: socket.socket | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    allowed_count: int = field(default=0, init=False)
    denied: list[str] = field(default_factory=list, init=False)

    def _record(self, verdict: str, target: str) -> None:
        with self._lock:
            if verdict == "allow":
                self.allowed_count += 1
            elif target not in self.denied:
                self.denied.append(target)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{verdict} {target}\n")

    def start(self) -> str:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_host, self.port))
        server.listen(64)
        self.port = server.getsockname()[1]
        self._server = server
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return f"http://{self.bind_host}:{self.port}"

    def _serve(self) -> None:
        assert self._server is not None
        self._server.settimeout(1.0)
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _connect_out(self, host: str, port: int) -> socket.socket:
        """Reach one allowed HTTPS target, directly or through an upstream proxy.

        Set AI4AI_UPSTREAM_PROXY, AI4AI_EGRESS_UPSTREAM, or HTTPS_PROXY to chain.
        The local allowlist is enforced before the upstream is contacted.
        """

        # Keep both project-specific names for compatibility with existing deployments.
        upstream_url = (
            os.environ.get("AI4AI_UPSTREAM_PROXY")
            or os.environ.get("AI4AI_EGRESS_UPSTREAM")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or ""
        ).strip()
        if not upstream_url:
            print(
                "egress: no upstream proxy set "
                "(AI4AI_UPSTREAM_PROXY / AI4AI_EGRESS_UPSTREAM / HTTPS_PROXY); "
                f"connecting directly to {host}:{port}",
                flush=True,
            )
            return socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)

        parsed = urllib.parse.urlsplit(
            upstream_url if "://" in upstream_url else f"http://{upstream_url}"
        )
        if not parsed.hostname:
            raise OSError(f"upstream proxy {upstream_url!r} has no host")
        sock = socket.create_connection(
            (parsed.hostname, parsed.port or 3128), timeout=CONNECT_TIMEOUT
        )
        try:
            sock.sendall(
                f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()
            )
            sock.settimeout(CONNECT_TIMEOUT)
            response = b""
            while b"\r\n\r\n" not in response:
                chunk = sock.recv(RELAY_CHUNK)
                if not chunk:
                    raise OSError("upstream proxy closed during CONNECT")
                response += chunk
                if len(response) > MAX_REQUEST_BYTES:
                    raise OSError("upstream proxy sent an oversized CONNECT reply")
            status = response.split(b"\r\n", 1)[0].decode("latin-1", "replace")
            if " 200" not in status:
                # The upstream's own allowlist rejecting the host looks identical to a
                # dead upstream unless the status is reported.
                raise OSError(f"upstream proxy refused {host}:{port}: {status}")
            return sock
        except OSError:
            sock.close()
            raise

    def _target_allowed(self, host: str, port: int) -> bool:
        target = host if port == 443 else f"{host}:{port}"
        return target in self.allowed_hosts

    def _handle(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            client.settimeout(CONNECT_TIMEOUT)
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = client.recv(RELAY_CHUNK)
                if not chunk:
                    return
                request += chunk
                if len(request) > MAX_REQUEST_BYTES:
                    client.sendall(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                    return

            line = request.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT":
                self._record("deny", f"non-connect:{line[:60]}")
                client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                return

            host, _, port_text = parts[1].rpartition(":")
            if not host or not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
                self._record("deny", parts[1])
                client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            port = int(port_text)
            if not self._target_allowed(host, port):
                self._record("deny", parts[1])
                client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return

            try:
                upstream = self._connect_out(host, port)
            except OSError:
                self._record("deny", f"unreachable:{host}:{port}")
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                return

            self._record("allow", parts[1])
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._relay(client, upstream)
        except OSError:
            return
        finally:
            for sock in (client, upstream):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    @staticmethod
    def _keepalive(sock: socket.socket) -> None:
        """Let the OS decide whether the peer is gone, instead of inferring it from silence.

        Without this the only liveness signal is bytes arriving, so a model that thinks for
        twenty minutes is indistinguishable from a crashed one. The per-socket options are
        Linux-only names; a platform without them still gets SO_KEEPALIVE with system
        defaults, which is a weaker version of the same thing rather than nothing.
        """

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            return
        for name, value in (
            ("TCP_KEEPIDLE", KEEPALIVE_IDLE_SEC),
            ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL_SEC),
            ("TCP_KEEPCNT", KEEPALIVE_FAILURES),
        ):
            option = getattr(socket, name, None)
            if option is None:
                continue
            try:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
            except OSError:
                pass

    def _relay(self, left: socket.socket, right: socket.socket) -> None:
        left.settimeout(None)
        right.settimeout(None)
        self._keepalive(left)
        self._keepalive(right)
        with selectors.DefaultSelector() as selector:
            selector.register(left, selectors.EVENT_READ, right)
            selector.register(right, selectors.EVENT_READ, left)
            while not self._stop.is_set():
                events = selector.select(timeout=IDLE_TIMEOUT)
                if not events:
                    self._record("deny", "idle-timeout")
                    return
                for key, _ in events:
                    source: socket.socket = key.fileobj  # type: ignore[assignment]
                    target: socket.socket = key.data
                    try:
                        data = source.recv(RELAY_CHUNK)
                    except OSError:
                        return
                    if not data:
                        return
                    try:
                        target.sendall(data)
                    except OSError:
                        return

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)


def _docker(*args: str, runtime: str = DEFAULT_RUNTIME) -> str:
    result = subprocess.run(
        [*runtime_argv(runtime), *args], check=False, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise EgressError(f"{runtime} {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def create_network(name: str, *, runtime: str = DEFAULT_RUNTIME) -> str:
    """Create an --internal bridge and return the host-side gateway address.

    --internal is what makes this safe to point a proxy at: the container gets no
    route to anything except the gateway, and no DNS.
    """

    existing = subprocess.run(
        [*runtime_argv(runtime), "network", "inspect", name, "--format", "{{.Internal}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if existing.returncode == 0:
        if existing.stdout.strip() != "true":
            raise EgressError(f"network {name} already exists and is not --internal")
    else:
        _docker("network", "create", "--internal", name, runtime=runtime)

    gateway = _docker(
        "network",
        "inspect",
        name,
        "--format",
        "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
        runtime=runtime,
    )
    if not gateway:
        raise EgressError(f"network {name} reported no gateway")
    return gateway


def remove_network(name: str, *, runtime: str = DEFAULT_RUNTIME) -> None:
    subprocess.run(
        [*runtime_argv(runtime), "network", "rm", name],
        check=False,
        capture_output=True,
        timeout=60,
    )
