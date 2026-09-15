import pytest
import os
import sys
import filecmp
import tempfile
from unittest.mock import patch
from io import StringIO

from conin.util import try_import
from conin.common.unified import load_model
from conin.common.bin.convert_pgm import main

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite

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


class TestConvertPGMBasic:
    """Test basic functionality and error handling"""

    def test_no_arguments(self):
        """Test that running with no arguments shows an error"""
        exit_code, stdout, stderr = run_convert_pgm([])
        assert exit_code != 0

    def test_help(self):
        """Test help flag"""
        exit_code, stdout, stderr = run_convert_pgm(["--help"])
        assert exit_code == 0
        assert "convert graphical model" in stdout.lower()

    def test_missing_input_file(self):
        """Test error when input file doesn't exist"""
        exit_code, stdout, stderr = run_convert_pgm(
            ["nonexistent.uai", "--uai", "output.uai"]
        )
        assert exit_code != 0
        assert "not found" in stderr.lower()

    def test_no_output_format(self):
        """Test error when no output format is specified"""
        input_file = os.path.join(cwd, "asia.uai")
        exit_code, stdout, stderr = run_convert_pgm([input_file])
        assert exit_code != 0
        assert "output format" in stderr.lower()


class TestConvertPGMUAI:
    """Test UAI format conversions"""

    def test_uai_to_uai(self):
        """Test converting UAI to UAI (identity conversion)"""
        input_file = os.path.join(cwd, "asia.uai")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the file was created and can be loaded
            pgm = load_model(output_file)
            assert pgm is not None
            assert len(pgm.nodes) > 0

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_uai_quiet_mode(self):
        """Test quiet mode suppresses output"""
        input_file = os.path.join(cwd, "asia.uai")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert stdout == ""  # No output in quiet mode

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_uai_without_extension(self):
        """Test that .uai extension is added automatically"""
        input_file = os.path.join(cwd, "asia.uai")

        with tempfile.NamedTemporaryFile(suffix="_test", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file]
            )
            assert exit_code == 0
            # The tool should add .uai extension
            expected_file = output_file + ".uai"
            assert os.path.exists(expected_file)

        finally:
            expected_file = output_file + ".uai"
            if os.path.exists(expected_file):
                os.remove(expected_file)
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_compressed_uai_to_uai(self):
        """Test converting compressed UAI to UAI"""
        input_file = os.path.join(cwd, "asia_compressed.uai.gz")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the output matches the uncompressed version
            pgm_output = load_model(output_file)
            pgm_orig = load_model(os.path.join(cwd, "asia.uai"))

            assert len(pgm_output.nodes) == len(pgm_orig.nodes)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_large_model_barley(self):
        """Test converting a larger model (barley)"""
        input_file = os.path.join(cwd, "barley.uai")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the model has the expected size
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 10  # Barley is a large model

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
class TestConvertPGMBIF:
    """Test BIF format conversions"""

    def test_markov_network_to_bif_error(self):
        """Test that converting Markov network to BIF raises an error"""
        input_file = os.path.join(cwd, "cancer_mn.uai")

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--bif", output_file, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            # Check that the error message is informative
            assert len(stderr) > 0

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_deer_markov_network_to_bif_error(self):
        """Test that converting deer Markov network to BIF raises an error"""
        input_file = os.path.join(cwd, "deer.uai")

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--bif", output_file, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            assert len(stderr) > 0

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_bif_to_uai(self):
        """Test converting BIF to UAI"""
        input_file = os.path.join(cwd, "asia.bif")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify the model was converted correctly
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 0

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_bif(self):
        """Test converting UAI to BIF"""
        input_file = os.path.join(cwd, "cancer_bn.uai")

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--bif", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify the model can be loaded back
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 0

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_bif_to_bif(self):
        """Test converting BIF to BIF (identity conversion)"""
        input_file = os.path.join(cwd, "asia.bif")

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--bif", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify both models have the same structure
            pgm_input = load_model(input_file)
            pgm_output = load_model(output_file)
            assert len(pgm_input.nodes) == len(pgm_output.nodes)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_compressed_bif_to_uai(self):
        """Test converting compressed BIF to UAI"""
        input_file = os.path.join(cwd, "asia_compressed.bif.gz")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
class TestConvertPGMMultipleFormats:
    """Test converting to multiple formats simultaneously"""

    def test_markov_network_to_both_formats_error(self):
        """Test that converting Markov network to both formats fails for BIF"""
        input_file = os.path.join(cwd, "cancer_mn.uai")

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            assert len(stderr) > 0

        finally:
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)

    def test_uai_to_both_formats(self):
        """Test converting UAI to both UAI and BIF"""
        input_file = os.path.join(cwd, "cancer_bn.uai")

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(uai_output)
            assert os.path.exists(bif_output)

            # Verify both outputs are valid
            pgm_uai = load_model(uai_output)
            pgm_bif = load_model(bif_output)
            assert len(pgm_uai.nodes) == len(pgm_bif.nodes)

        finally:
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)

    def test_bif_to_both_formats(self):
        """Test converting BIF to both UAI and BIF"""
        input_file = os.path.join(cwd, "asia.bif")

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(uai_output)
            assert os.path.exists(bif_output)

            # Verify both outputs have the same structure
            pgm_uai = load_model(uai_output)
            pgm_bif = load_model(bif_output)
            assert len(pgm_uai.nodes) == len(pgm_bif.nodes)

        finally:
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)


class TestConvertPGMRoundTrip:
    """Test round-trip conversions preserve model structure"""

    def test_cancer_bn_roundtrip(self):
        """Test UAI -> save -> load produces identical model"""
        input_file = os.path.join(cwd, "cancer_bn.uai")

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            # Convert UAI to UAI
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # Load both models and compare
            pgm_orig = load_model(input_file)
            pgm_output = load_model(output_file)

            # Check same number of nodes
            assert len(pgm_orig.nodes) == len(pgm_output.nodes)

            # Check same number of CPDs
            assert len(pgm_orig.cpds) == len(pgm_output.cpds)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    @pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
    def test_bif_uai_bif_roundtrip(self):
        """Test BIF -> UAI -> BIF preserves structure"""
        input_file = os.path.join(cwd, "asia.bif")

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_file = tmp_uai.name
            bif_file = tmp_bif.name

        try:
            # BIF -> UAI
            exit_code, stdout, stderr = run_convert_pgm(
                [input_file, "--uai", uai_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # UAI -> BIF
            exit_code, stdout, stderr = run_convert_pgm(
                [uai_file, "--bif", bif_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # Compare original and final
            pgm_orig = load_model(input_file)
            pgm_final = load_model(bif_file)

            assert len(pgm_orig.nodes) == len(pgm_final.nodes)

        finally:
            if os.path.exists(uai_file):
                os.remove(uai_file)
            if os.path.exists(bif_file):
                os.remove(bif_file)
