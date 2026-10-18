"""Render every UI view against REAL payloads from the backend service layer (needs Node.js).
Run from the project root:  python scripts/check_frontend.py
Checks: no undefined/NaN leaks, expected sections present, user-supplied text is HTML-escaped.
It does NOT click through the UI in a browser."""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
node = shutil.which("node")
if not node:
    sys.exit("Node.js not found; install Node.js or skip this optional check.")
tmp = Path(tempfile.mkdtemp())
js = re.search(r"<script>(.*)</script>", (root / "frontend" / "index.html").read_text(encoding="utf-8"), re.S).group(1)
(tmp / "app.js").write_text(js, encoding="utf-8")
subprocess.run([sys.executable, str(root / "scripts" / "_gen.py"), str(tmp / "payloads.json")], cwd=root, check=True)
r = subprocess.run([node, "--check", str(tmp / "app.js")])
r = subprocess.run([node, str(root / "scripts" / "frontend_view_tests.js"), str(tmp / "app.js"), str(tmp / "payloads.json")])
sys.exit(r.returncode)
