"""Exercise HTTP delivery independently of an IPython kernel message channel."""

import importlib.util
from http.client import HTTPResponse
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import quote
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab_downloads.py"
SPEC = importlib.util.spec_from_file_location("colab_downloads", SCRIPT)
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


class MemoryConnection:
    """Supply HTTP bytes to the real handler without opening a network socket."""

    def __init__(self, request):
        self.input = BytesIO(request)
        self.output = BytesIO()

    def makefile(self, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class ColabDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "result with spaces.zip"
        with ZipFile(self.archive, "w") as archive:
            archive.writestr("test.txt", "HTTP transport fixture")
        (self.directory / "unfinished.zip.tmp").write_bytes(b"partial")

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, path):
        connection = MemoryConnection(f"GET {path} HTTP/1.0\r\n\r\n".encode("ascii"))
        downloads.DownloadHandler(
            connection, ("127.0.0.1", 0), SimpleNamespace(),
            directory=self.directory.resolve(), auto_download=False,
        )
        response = HTTPResponse(SimpleNamespace(
            makefile=lambda *args: BytesIO(connection.output.getvalue())))
        response.begin()
        return response

    def test_completed_zip_is_delivered_byte_for_byte_with_attachment_header(self):
        with self.request("/download/" + quote(self.archive.name)) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), self.archive.read_bytes())
            self.assertEqual(response.headers["Content-Type"], "application/zip")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertEqual(int(response.headers["Content-Length"]), self.archive.stat().st_size)

    def test_index_updates_when_a_new_completed_archive_appears(self):
        with self.request("/index.json") as response:
            before = json.load(response)
        self.assertEqual([item["name"] for item in before], [self.archive.name])
        (self.directory / "second.zip").write_bytes(self.archive.read_bytes())
        with self.request("/index.json") as response:
            after = json.load(response)
        self.assertEqual(len(after), 2)

    def test_other_paths_and_files_are_rejected(self):
        for path in ["/download/../test.txt", "/download/%2Fetc%2Fpasswd", "/download/unfinished.zip.tmp"]:
            with self.subTest(path=path), self.request(path) as response:
                self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
