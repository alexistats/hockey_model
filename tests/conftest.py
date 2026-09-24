"""Shared fixtures."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parent.parent / "ui" / "draft_room.html"


def page_block(name: str) -> str:
    """A self-contained block of the draft page's script, between its markers.

    The page carries its own copies of logic the API already has - the room
    model, the schedule fit - because it runs with no server behind it. Each
    copy sits between `/* name:begin` and `/* name:end */` so a test can run it
    on its own and hold it to the Python.
    """
    html = PAGE.read_text(encoding="utf-8")
    start, end = html.index(f"/* {name}:begin"), html.index(f"/* {name}:end */")
    return html[start:end]


@pytest.fixture
def page_source():
    """`page_block`, for tests that check a block's text rather than run it."""
    return page_block


@pytest.fixture
def page_js(tmp_path):
    """Run one of the page's blocks under Node: page_js(block, expression, given).

    `given` is passed in as `x`, and the expression's value comes back as JSON.
    Skips when Node is not installed.
    """
    if shutil.which("node") is None:
        pytest.skip("needs Node to run the page's JavaScript")

    def run(block: str, expression: str, given: dict):
        script = tmp_path / f"{block}.js"
        script.write_text(
            page_block(block)
            + "\nconst x = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
            + f"process.stdout.write(JSON.stringify({expression}));\n",
            encoding="utf-8",
        )
        ran = subprocess.run(
            ["node", str(script)], input=json.dumps(given), capture_output=True, text=True
        )
        assert ran.returncode == 0, ran.stderr
        return json.loads(ran.stdout)

    return run
