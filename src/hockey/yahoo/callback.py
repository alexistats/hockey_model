"""A throwaway HTTPS listener that catches Yahoo's redirect.

Yahoo redirects to the app's registered URI with the authorization code in the
query string. Without something listening there the browser shows a connection
error and the code has to be read out of the address bar, which is fiddly and
easy to get wrong - and the code expires in about a minute, so a fumbled copy
means starting over.

This runs a one-shot server on the registered port, serves a self-signed
certificate, captures the code, and shuts down. The browser will warn about the
certificate because it is self-signed; that warning is expected and safe here.
Nothing but the local browser ever connects, the certificate lives in a
temporary directory, and the process exits as soon as the code arrives.
"""

import http.server
import logging
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

PAGE = """<!doctype html><meta charset="utf-8">
<title>Authorized</title>
<body style="font:16px system-ui;max-width:34rem;margin:4rem auto;color:#222">
<h2 style="color:#137333">Authorization received</h2>
<p>You can close this tab and go back to the terminal.</p>
</body>"""

ERROR_PAGE = """<!doctype html><meta charset="utf-8">
<title>No code</title>
<body style="font:16px system-ui;max-width:34rem;margin:4rem auto;color:#222">
<h2 style="color:#c5221f">No authorization code in that request</h2>
<p>Yahoo redirected here without a <code>code</code> parameter. Check the
terminal for what it sent instead.</p>
</body>"""


class CallbackError(RuntimeError):
    """The redirect arrived without a usable authorization code."""


def find_openssl() -> str:
    """The openssl executable, hunted rather than assumed.

    Git for Windows ships one, but only Git Bash puts it on PATH - from
    PowerShell or cmd the bare name fails with a WinError 2 that names no file,
    which is a poor way to learn that a certificate could not be made. So look
    on PATH first, then in the places Git installs it, then alongside git
    itself if it landed somewhere unusual.
    """
    found = shutil.which("openssl")
    if found:
        return found

    candidates = [
        Path(base) / sub / "bin" / "openssl.exe"
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        )
        if base
        for sub in ("Git/usr", "Git/mingw64", "Programs/Git/usr", "Programs/Git/mingw64")
    ]
    git = shutil.which("git")
    if git:
        # .../Git/cmd/git.exe -> .../Git/usr/bin/openssl.exe
        root = Path(git).parent.parent
        candidates += [
            root / "usr" / "bin" / "openssl.exe",
            root / "mingw64" / "bin" / "openssl.exe",
        ]

    for path in candidates:
        if path.is_file():
            logger.debug("using openssl at %s", path)
            return str(path)

    raise CallbackError(
        "No openssl executable found, so the local HTTPS listener cannot be given a "
        "certificate. Git for Windows ships one, but only Git Bash puts it on PATH.\n\n"
        "Either run this from Git Bash, or authorize without the listener:\n\n"
        "    python -m hockey.yahoo auth-url\n\n"
        "which prints a URL to open, then hand the code back with "
        "`python -m hockey.yahoo exchange <code>` within about a minute."
    )


def make_self_signed_cert(directory: Path) -> tuple[Path, Path]:
    """A localhost certificate, via the openssl that ships with Git for Windows.

    Self-signed on purpose: the only client is the browser on this machine,
    for a few seconds, and the alternative is asking the user to transcribe a
    short-lived code out of an error page.
    """
    key = directory / "key.pem"
    cert = directory / "cert.pem"
    result = subprocess.run(
        [
            find_openssl(),
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not cert.is_file() or not key.is_file():
        raise CallbackError(
            f"openssl could not make a certificate for the local listener "
            f"(exit {result.returncode}): {result.stderr.strip()[:400]}\n\n"
            f"Authorize without the listener instead:\n\n"
            f"    python -m hockey.yahoo auth-url"
        )
    return cert, key


class _Handler(http.server.BaseHTTPRequestHandler):
    captured: dict = {}

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        params = parse_qs(urlparse(self.path).query)
        code = params.get("code", [None])[0]
        if code:
            _Handler.captured["code"] = code
            body = PAGE
        else:
            _Handler.captured["error"] = params
            body = ERROR_PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass  # the query string holds the code; keep it out of the log


class _Server(http.server.HTTPServer):
    """Keeps serving through a failed connection.

    A browser opening a self-signed HTTPS page does not make one clean
    request. It may preconnect and abandon, fail the handshake while the user
    is still reading the certificate warning, then connect properly once they
    click through. Treating the first connection as the only one meant the
    listener gave up seconds after starting, before the user had done
    anything.
    """

    def handle_error(self, request, client_address):
        pass  # handshake failures before the user clicks through are expected


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def capture_code(redirect_uri: str, authorize_url: str, timeout: float = 300.0) -> str:
    """Open the authorize URL and return the code Yahoo redirects back with."""
    parsed = urlparse(redirect_uri)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not port_is_free(port):
        raise CallbackError(
            f"something is already listening on port {port}, which is the port the "
            f"Yahoo app redirects to. Stop it and try again, or authorize manually "
            f"with `python -m hockey.yahoo auth-url`."
        )

    _Handler.captured = {}
    with tempfile.TemporaryDirectory() as tmp:
        server = _Server(("127.0.0.1", port), _Handler)
        if parsed.scheme == "https":
            cert, key = make_self_signed_cert(Path(tmp))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=cert, keyfile=key)
            server.socket = context.wrap_socket(server.socket, server_side=True)

        # serve_forever, not handle_request: the browser's first connection to
        # a self-signed page is usually a failed handshake while the warning is
        # still on screen. Handling exactly one request meant giving up
        # seconds after starting, before the user had clicked anything.
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2})
        thread.daemon = True
        thread.start()

        print()
        print("Opening your browser to Yahoo. If it does not open, paste this in:")
        print()
        print(f"   {authorize_url}")
        print()
        print(f"Waiting on {redirect_uri} for the redirect...")
        if parsed.scheme == "https":
            print()
            print("Your browser will warn that the certificate is not trusted. That is")
            print("expected - the certificate is generated here, used by this machine")
            print("only, and thrown away. Choose Advanced, then proceed to localhost.")
        print()
        webbrowser.open(authorize_url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not _Handler.captured:
            time.sleep(0.2)
        server.shutdown()
        thread.join(5)
        server.server_close()

    if "code" in _Handler.captured:
        return _Handler.captured["code"]
    if "error" in _Handler.captured:
        raise CallbackError(f"Yahoo redirected without a code: {_Handler.captured['error']}")
    raise CallbackError(
        f"nothing arrived on {redirect_uri} within {timeout:.0f} seconds. If the "
        f"browser never reached a Yahoo consent screen, the authorize URL is being "
        f"rejected - run `python -m hockey.yahoo auth-url` and open it by hand to "
        f"see what Yahoo says."
    )
