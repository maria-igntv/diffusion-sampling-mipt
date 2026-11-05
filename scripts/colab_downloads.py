"""Download ZIPs from a background HTTP server through Colab's port iframe.

Upload this standalone file into an existing Colab session to recover archives.
The server binds to loopback and serves ZIPs from one selected directory.
"""

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from threading import Thread
from urllib.parse import quote, unquote, urlsplit


PAGE = """<!doctype html>
<html lang="ru"><meta charset="utf-8">
<title>Архивы результатов</title>
<style>
body{font:14px system-ui,sans-serif;margin:16px;color:#17202a}
button,a{display:inline-block;padding:8px 12px;border:1px solid #bcc8d4;
border-radius:5px;background:#f7fafc;color:#174b87;text-decoration:none;cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:12px}
td{padding:8px;border-bottom:1px solid #dce3e9;overflow-wrap:anywhere}
.size{white-space:nowrap;color:#526171}label{margin-left:12px}
</style>
<h3>Архивы результатов</h3>
<button id="refresh">Обновить список</button>
<label><input id="auto" type="checkbox" __AUTO_CHECKED__> Скачивать новые архивы</label>
<p id="status">Загрузка списка…</p><table><tbody id="files"></tbody></table>
<script>
let initialized = false;
const seen = new Set();
async function refresh() {
  try {
    const response = await fetch('index.json', {cache: 'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const files = await response.json();
    const body = document.getElementById('files');
    body.replaceChildren();
    for (const file of files) {
      const row = document.createElement('tr');
      const name = document.createElement('td');
      name.textContent = file.name;
      const size = document.createElement('td');
      size.className = 'size';
      size.textContent = (file.bytes / 1048576).toFixed(1) + ' MiB';
      const action = document.createElement('td');
      const link = document.createElement('a');
      link.textContent = 'Скачать';
      link.href = file.href;
      link.download = file.name;
      action.appendChild(link);
      row.append(name, size, action);
      body.appendChild(row);
      if (initialized && !seen.has(file.name) && document.getElementById('auto').checked) {
        link.click();
      }
      seen.add(file.name);
    }
    initialized = true;
    document.getElementById('status').textContent =
      'Готовых архивов ' + files.length + '. Нажмите «Скачать» у нужного файла.';
  } catch (error) {
    document.getElementById('status').textContent = 'Ошибка соединения — ' + error.message;
  }
}
document.getElementById('refresh').onclick = refresh;
refresh();
setInterval(refresh, 5000);
</script></html>"""


class DownloadHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, auto_download, **kwargs):
        self.directory = directory
        self.page = PAGE.replace("__AUTO_CHECKED__", "checked" if auto_download else "").encode("utf-8")
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def _headers(self, content_type, size):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("x-colab-notebook-cache-control", "no-cache")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._headers("text/html; charset=utf-8", len(self.page))
            self.end_headers()
            self.wfile.write(self.page)
            return
        if path == "/index.json":
            records = []
            for archive in sorted(self.directory.glob("*.zip")):
                resolved = archive.resolve()
                if resolved.parent != self.directory or not resolved.is_file():
                    continue
                records.append({"name": archive.name, "bytes": archive.stat().st_size,
                                "href": "download/" + quote(archive.name, safe="")})
            data = json.dumps(records, ensure_ascii=False).encode("utf-8")
            self._headers("application/json; charset=utf-8", len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/download/"):
            name = unquote(path[len("/download/"):])
            archive = (self.directory / name).resolve()
            if not name or Path(name).name != name or not name.endswith(".zip") or archive.parent != self.directory or not archive.is_file():
                self.send_error(404)
                return
            try:
                with archive.open("rb") as stream:
                    self._headers("application/zip", archive.stat().st_size)
                    self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(name, safe=""))
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)


def start_download_server(directory, auto_download=False, display_panel=True):
    """Return a running server. Use server.shutdown()/server_close() to stop it."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Archive directory does not exist: {directory}")
    if display_panel:
        from google.colab import output
    handler = partial(DownloadHandler, directory=directory, auto_download=auto_download)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True, name="colab-zip-downloads").start()
    if display_panel:
        output.serve_kernel_port_as_iframe(server.server_port, height=420)
    return server
