"""Tests of the convert_pgm command-line tool.

The conversions themselves are load_model/save_model, which test_load_model.py
and test_save_model.py test; these tests check the command line: arguments,
output file names, messages and exit codes.
"""

import pytest
import os
import shutil
import sys
from io import StringIO

from conin.algorithms.inference import map_query
from conin.io import load_model
from conin.io.convert_pgm import main
from conin.tests.optional import pgmpy_installed

needs_pgmpy = pytest.mark.skipif(not pgmpy_installed, reason="pgmpy not installed")

cwd = os.path.dirname(__file__)


def run_convert_pgm(args):
    """
    Helper function to run convert_pgm with command-line arguments.
    Returns (exit_code, stdout, stderr)
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv

    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        sys.argv = ["convert_pgm"] + args

        exit_code = 0
        try:
            main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0

        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()

        return exit_code, stdout, stderr

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.argv = old_argv


def copy_input(name, tmp_path):
    """Copy a test file into tmp_path (convert_pgm writes next to its input)."""
    path = str(tmp_path / ("test_" + name))
    shutil.copy(os.path.join(cwd, name), path)
    return path


#
# Arguments and errors
#


def test_no_arguments():
    exit_code, stdout, stderr = run_convert_pgm([])
    assert exit_code != 0


def test_help():
    exit_code, stdout, stderr = run_convert_pgm(["--help"])
    assert exit_code == 0
    assert "convert graphical model" in stdout.lower()


def test_missing_input_file(tmp_path):
    exit_code, stdout, stderr = run_convert_pgm(
        [str(tmp_path / "nonexistent.uai"), "--uai", str(tmp_path / "output.uai")]
    )
    assert exit_code != 0
    assert "not found" in stderr.lower()


def test_no_output_format(tmp_path):
    exit_code, stdout, stderr = run_convert_pgm([copy_input("asia.uai", tmp_path)])
    assert exit_code != 0
    assert "output format" in stderr.lower()


@needs_pgmpy
def test_markov_network_to_bif_error(tmp_path):
    output_file = str(tmp_path / "out.bif")
    exit_code, stdout, stderr = run_convert_pgm(
        [copy_input("cancer_mn.uai", tmp_path), "--bif", output_file, "--quiet"]
    )
    assert exit_code != 0
    assert "Cannot save Markov network to BIF format" in stderr
    assert not os.path.exists(output_file)


#
# Output file names
#


def test_uai_to_uai(tmp_path, assert_same_model):
    temp_input = copy_input("cancer_bn.uai", tmp_path)
    output_file = str(tmp_path / "out.uai")
    exit_code, stdout, stderr = run_convert_pgm([temp_input, "--uai", output_file])
    assert exit_code == 0
    assert f"Successfully saved to: {output_file}" in stdout
    assert_same_model(load_model(output_file), load_model(temp_input))


def test_quiet_mode(tmp_path):
    output_file = str(tmp_path / "out.uai")
    exit_code, stdout, stderr = run_convert_pgm(
        [copy_input("asia.uai", tmp_path), "--uai", output_file, "--quiet"]
    )
    assert exit_code == 0
    assert stdout == ""
    assert os.path.exists(output_file)


def test_output_without_extension_gets_one(tmp_path):
    output_file = str(tmp_path / "out_test")
    exit_code, stdout, stderr = run_convert_pgm(
        [copy_input("asia.uai", tmp_path), "--uai", output_file, "--quiet"]
    )
    assert exit_code == 0
    assert sorted(os.listdir(tmp_path)) == ["out_test.uai", "test_asia.uai"]


def test_compressed_uai_auto_filename(tmp_path, assert_same_model):
    # The .uai.gz suffix is replaced as a whole.
    temp_input = copy_input("asia_compressed.uai.gz", tmp_path)
    exit_code, stdout, stderr = run_convert_pgm([temp_input, "--uai", "--quiet"])
    assert exit_code == 0
    expected_output = str(tmp_path / "test_asia_compressed.uai")
    assert_same_model(load_model(expected_output), load_model(temp_input))


@needs_pgmpy
def test_bif_auto_filename(tmp_path):
    temp_input = copy_input("cancer_bn.uai", tmp_path)
    exit_code, stdout, stderr = run_convert_pgm([temp_input, "--bif", "--quiet"])
    assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
    output = load_model(str(tmp_path / "test_cancer_bn.bif"))
    original = load_model(temp_input)
    # BIF state names are strings, so compare the MAP log-probability.
    assert output.nodes == original.nodes
    assert map_query(
        output, method="variable_elimination"
    ).solution.log_potential == pytest.approx(
        map_query(original, method="variable_elimination").solution.log_potential
    )


@needs_pgmpy
def test_both_formats_auto_filename(tmp_path):
    temp_input = copy_input("asia.bif", tmp_path)
    exit_code, stdout, stderr = run_convert_pgm(
        [temp_input, "--uai", "--bif", "--quiet"]
    )
    assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
    # The BIF output overwrites the input copy (same name).
    assert sorted(os.listdir(tmp_path)) == ["test_asia.bif", "test_asia.uai"]
    assert len(load_model(str(tmp_path / "test_asia.uai")).nodes) == 8
    assert sorted(load_model(temp_input).nodes) == sorted(
        load_model(os.path.join(cwd, "asia.bif")).nodes
    )
