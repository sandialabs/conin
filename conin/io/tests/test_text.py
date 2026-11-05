"""Reading model files: gzipped files and unreadable text (conin.io.text)."""

import gzip
import os
import shutil

import pytest

import conin.examples.hidden_markov_model as hex_
from conin.exceptions import ModelFileError
from conin.io import load_cfn, load_json, load_model, load_uai, save_json
from conin.io.text import read_text

cwd = os.path.dirname(__file__)


def _gzip(source, target):
    with open(source, "rb") as f, gzip.open(target, "wb") as g:
        shutil.copyfileobj(f, g)


@pytest.mark.parametrize(
    "loader,name", [(load_uai, "deer.uai"), (load_cfn, "small.cfn")]
)
def test_direct_loaders_read_gzipped_files(loader, name, tmp_path, assert_same_model):
    # They opened every file as text, so a .gz file raised UnicodeDecodeError.
    source = os.path.join(cwd, name)
    compressed = tmp_path / f"{name}.gz"
    _gzip(source, compressed)
    assert_same_model(loader(filename=str(compressed)), loader(filename=source))
    assert_same_model(load_model(str(compressed)), loader(filename=source))


def test_load_json_reads_gzipped_files(tmp_path):
    hmm = hex_.create_hmm2()
    compressed = tmp_path / "hmm.json.gz"
    save_json(hmm, compressed)
    with gzip.open(compressed, "rt", encoding="utf-8") as f:
        assert f.read() == save_json(hmm)
    assert load_json(filename=compressed) == hmm
    assert load_json(filename=str(compressed)) == hmm


@pytest.mark.parametrize(
    "loader,suffix", [(load_uai, ".uai"), (load_cfn, ".cfn"), (load_json, ".json")]
)
def test_unreadable_files_raise_model_file_error(loader, suffix, tmp_path):
    not_utf8 = tmp_path / f"latin1{suffix}"
    not_utf8.write_bytes("MARKOV caf\xe9".encode("latin-1"))
    with pytest.raises(ModelFileError, match="Cannot read"):
        loader(filename=str(not_utf8))

    not_gzip = tmp_path / f"plain{suffix}.gz"
    not_gzip.write_text("MARKOV")
    with pytest.raises(ModelFileError, match="Cannot read"):
        loader(filename=str(not_gzip))


@pytest.mark.parametrize(
    "loader,message",
    [
        (load_uai, "Cannot read missing UAI file"),
        (load_cfn, "Cannot read missing CFN file"),
        (load_json, "Missing file"),
    ],
)
def test_missing_files(loader, message, tmp_path):
    with pytest.raises(FileNotFoundError, match=message):
        loader(filename=str(tmp_path / "missing.gz"))


def test_read_text(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_text("café\n", encoding="utf-8")
    compressed = tmp_path / "a.txt.gz"
    _gzip(plain, compressed)
    assert read_text(plain) == read_text(str(compressed)) == "café\n"
