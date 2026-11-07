"""LIAR download: HTTP errors must never be saved as data, every file must match
the release checksum, and a failed download must fail loudly."""

import hashlib
import io
import zipfile

import pytest
import requests

import download_data

FILES = {"train.tsv": b"train rows\n", "valid.tsv": b"valid rows\n", "test.tsv": b"test rows\n"}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.headers = {"content-length": str(len(body))}
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def liar_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "LIAR_SHA256",
                        {name: hashlib.sha256(body).hexdigest() for name, body in FILES.items()})
    return tmp_path / "liar"


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buf.getvalue()


def _serve(monkeypatch, route):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(*route(url))

    monkeypatch.setattr(download_data.requests, "get", fake_get)
    return calls


def _saved(liar_dir):
    return {p.name: p.read_bytes() for p in liar_dir.iterdir()}


def test_mirror_files_are_saved_with_a_timeout(liar_dir, monkeypatch):
    calls = _serve(monkeypatch, lambda url: (200, FILES[url.rsplit("/", 1)[1]]))

    download_data.download_liar()

    assert _saved(liar_dir) == FILES
    assert calls and all(kwargs.get("timeout") for kwargs in calls)


def test_error_page_is_not_saved_and_the_official_archive_is_used(liar_dir, monkeypatch):
    def route(url):
        if url == download_data.LIAR_ZIP:
            return 200, _zip(FILES)
        return 404, b"<html>404: Not Found</html>"

    _serve(monkeypatch, route)

    download_data.download_liar()

    assert _saved(liar_dir) == FILES


def test_checksum_mismatch_raises_and_leaves_no_files(liar_dir, monkeypatch):
    tampered = {name: b"tampered\n" for name in FILES}
    _serve(monkeypatch, lambda url: (200, _zip(tampered) if url == download_data.LIAR_ZIP else b"tampered\n"))

    with pytest.raises(ValueError, match="SHA-256"):
        download_data.download_liar()

    assert _saved(liar_dir) == {}


def test_corrupt_existing_file_is_downloaded_again(liar_dir, monkeypatch):
    liar_dir.mkdir()
    (liar_dir / "train.tsv").write_bytes(b"truncated")
    _serve(monkeypatch, lambda url: (200, FILES[url.rsplit("/", 1)[1]]))

    download_data.download_liar()

    assert _saved(liar_dir) == {**FILES, "train.tsv.bad": b"truncated"}


def test_corrupt_existing_file_is_moved_aside_even_if_the_download_fails(liar_dir, monkeypatch):
    liar_dir.mkdir()
    (liar_dir / "train.tsv").write_bytes(b"truncated")
    _serve(monkeypatch, lambda url: (404, b"not found"))

    with pytest.raises(requests.HTTPError):
        download_data.download_liar()

    assert _saved(liar_dir) == {"train.tsv.bad": b"truncated"}


def test_command_exits_non_zero_when_the_download_fails(liar_dir, monkeypatch):
    _serve(monkeypatch, lambda url: (404, b"not found"))
    monkeypatch.setattr("sys.argv", ["download_data.py", "--dataset", "liar"])

    assert download_data.main() == 1
