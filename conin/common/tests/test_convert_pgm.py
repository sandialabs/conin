import pytest
import os
import sys
import filecmp
import tempfile
import shutil
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


def make_temp_copy(source_file):
    """
    Create a temporary copy of a file for testing.
    Returns (temp_file_path, cleanup_function)
    """
    # Get the filename and determine extension
    basename = os.path.basename(source_file)

    # Create temp file with same basename in temp directory
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "test_" + basename)

    # Ensure the temp file doesn't already exist
    counter = 0
    original_temp_path = temp_path
    while os.path.exists(temp_path):
        counter += 1
        name_parts = original_temp_path.rsplit(".", 1)
        if len(name_parts) == 2:
            temp_path = f"{name_parts[0]}_{counter}.{name_parts[1]}"
        else:
            temp_path = f"{original_temp_path}_{counter}"

    # Copy source to temp
    shutil.copy(source_file, temp_path)

    def cleanup():
        """Remove temp file and any generated outputs"""
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Also remove potential output files
        base = os.path.splitext(temp_path)[0]
        if base.endswith(".uai") or base.endswith(".bif"):
            base = os.path.splitext(base)[0]
        for ext in [".uai", ".bif"]:
            output_file = base + ext
            if os.path.exists(output_file):
                os.remove(output_file)

    return temp_path, cleanup


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
        source_file = os.path.join(cwd, "asia.uai")
        temp_input, cleanup = make_temp_copy(source_file)

        try:
            exit_code, stdout, stderr = run_convert_pgm([temp_input])
            assert exit_code != 0
            assert "output format" in stderr.lower()
        finally:
            cleanup()


class TestConvertPGMUAI:
    """Test UAI format conversions"""

    def test_uai_auto_filename(self):
        """Test auto-generated filename for UAI output"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup = make_temp_copy(source_file)

        try:
            exit_code, stdout, stderr = run_convert_pgm([temp_input, "--uai"])
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # Calculate expected output filename
            base = os.path.splitext(temp_input)[0]
            expected_output = base + ".uai"
            assert os.path.exists(expected_output)

            # Verify the file was created and can be loaded
            pgm = load_model(expected_output)
            assert pgm is not None
            assert len(pgm.nodes) > 0

        finally:
            cleanup()

    def test_compressed_uai_auto_filename(self):
        """Test auto-generated filename from compressed input"""
        source_file = os.path.join(cwd, "asia_compressed.uai.gz")
        temp_input, cleanup = make_temp_copy(source_file)

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", "--quiet"]
            )
            assert exit_code == 0

            # Calculate expected output filename (strip .gz and replace with .uai)
            base = temp_input.replace(".uai.gz", "")
            expected_output = base + ".uai"
            assert os.path.exists(expected_output)

            # Verify the file was created
            pgm = load_model(expected_output)
            assert len(pgm.nodes) > 0

        finally:
            cleanup()

    def test_uai_to_uai(self):
        """Test converting UAI to UAI (identity conversion)"""
        source_file = os.path.join(cwd, "asia.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the file was created and can be loaded
            pgm = load_model(output_file)
            assert pgm is not None
            assert len(pgm.nodes) > 0

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_uai_quiet_mode(self):
        """Test quiet mode suppresses output"""
        source_file = os.path.join(cwd, "asia.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert stdout == ""  # No output in quiet mode

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_uai_without_extension(self):
        """Test that .uai extension is added automatically"""
        source_file = os.path.join(cwd, "asia.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix="_test", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file]
            )
            assert exit_code == 0
            # The tool should add .uai extension
            expected_file = output_file + ".uai"
            assert os.path.exists(expected_file)

        finally:
            cleanup_input()
            expected_file = output_file + ".uai"
            if os.path.exists(expected_file):
                os.remove(expected_file)
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_compressed_uai_to_uai(self):
        """Test converting compressed UAI to UAI"""
        source_file = os.path.join(cwd, "asia_compressed.uai.gz")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the output matches the uncompressed version
            pgm_output = load_model(output_file)
            pgm_orig = load_model(os.path.join(cwd, "asia.uai"))

            assert len(pgm_output.nodes) == len(pgm_orig.nodes)

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_large_model_barley(self):
        """Test converting a larger model (barley)"""
        source_file = os.path.join(cwd, "barley.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

            # Verify the model has the expected size
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 10  # Barley is a large model

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
class TestConvertPGMBIF:
    """Test BIF format conversions"""

    def test_bif_auto_filename(self):
        """Test auto-generated filename for BIF output"""
        source_file = os.path.join(cwd, "cancer_bn.uai")
        temp_input, cleanup = make_temp_copy(source_file)

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--bif", "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # Calculate expected output filename
            base = os.path.splitext(temp_input)[0]
            expected_output = base + ".bif"
            assert os.path.exists(expected_output)

            # Verify the file was created and can be loaded
            pgm = load_model(expected_output)
            assert pgm is not None
            assert len(pgm.nodes) > 0

        finally:
            cleanup()

    def test_both_formats_auto_filename(self):
        """Test auto-generated filenames for both output formats"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup = make_temp_copy(source_file)

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", "--bif", "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"

            # Calculate expected output filenames
            base = os.path.splitext(temp_input)[0]
            expected_uai = base + ".uai"
            expected_bif = base + ".bif"

            assert os.path.exists(expected_uai)
            # BIF output overwrites the temp input copy (same extension)
            assert os.path.exists(expected_bif)

            # Verify both files can be loaded
            pgm_uai = load_model(expected_uai)
            assert len(pgm_uai.nodes) > 0

        finally:
            cleanup()

    def test_markov_network_to_bif_error(self):
        """Test that converting Markov network to BIF raises an error"""
        source_file = os.path.join(cwd, "cancer_mn.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--bif", output_file, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            # Check that the error message is informative
            assert len(stderr) > 0

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_deer_markov_network_to_bif_error(self):
        """Test that converting deer Markov network to BIF raises an error"""
        source_file = os.path.join(cwd, "deer.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--bif", output_file, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            assert len(stderr) > 0

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_bif_to_uai(self):
        """Test converting BIF to UAI"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify the model was converted correctly
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 0

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_uai_to_bif(self):
        """Test converting UAI to BIF"""
        source_file = os.path.join(cwd, "cancer_bn.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--bif", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify the model can be loaded back
            pgm = load_model(output_file)
            assert len(pgm.nodes) > 0

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_bif_to_bif(self):
        """Test converting BIF to BIF (identity conversion)"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--bif", output_file, "--quiet"]
            )
            assert exit_code == 0, f"Unexpected failure {stdout=} {stderr=}"
            assert os.path.exists(output_file)

            # Verify both models have the same structure
            pgm_input = load_model(temp_input)
            pgm_output = load_model(output_file)
            assert len(pgm_input.nodes) == len(pgm_output.nodes)

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_compressed_bif_to_uai(self):
        """Test converting compressed BIF to UAI"""
        source_file = os.path.join(cwd, "asia_compressed.bif.gz")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(output_file)

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
class TestConvertPGMMultipleFormats:
    """Test converting to multiple formats simultaneously"""

    def test_markov_network_to_both_formats_error(self):
        """Test that converting Markov network to both formats fails for BIF"""
        source_file = os.path.join(cwd, "cancer_mn.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            # Should fail because Markov networks cannot be written to BIF
            assert exit_code != 0
            assert len(stderr) > 0

        finally:
            cleanup_input()
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)

    def test_uai_to_both_formats(self):
        """Test converting UAI to both UAI and BIF"""
        source_file = os.path.join(cwd, "cancer_bn.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(uai_output)
            assert os.path.exists(bif_output)

            # Verify both outputs are valid
            pgm_uai = load_model(uai_output)
            pgm_bif = load_model(bif_output)
            assert len(pgm_uai.nodes) == len(pgm_bif.nodes)

        finally:
            cleanup_input()
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)

    def test_bif_to_both_formats(self):
        """Test converting BIF to both UAI and BIF"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(
            suffix=".uai", delete=False
        ) as tmp_uai, tempfile.NamedTemporaryFile(
            suffix=".bif", delete=False
        ) as tmp_bif:
            uai_output = tmp_uai.name
            bif_output = tmp_bif.name

        try:
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", uai_output, "--bif", bif_output, "--quiet"]
            )
            assert exit_code == 0
            assert os.path.exists(uai_output)
            assert os.path.exists(bif_output)

            # Verify both outputs have the same structure
            pgm_uai = load_model(uai_output)
            pgm_bif = load_model(bif_output)
            assert len(pgm_uai.nodes) == len(pgm_bif.nodes)

        finally:
            cleanup_input()
            if os.path.exists(uai_output):
                os.remove(uai_output)
            if os.path.exists(bif_output):
                os.remove(bif_output)


class TestConvertPGMRoundTrip:
    """Test round-trip conversions preserve model structure"""

    def test_cancer_bn_roundtrip(self):
        """Test UAI -> save -> load produces identical model"""
        source_file = os.path.join(cwd, "cancer_bn.uai")
        temp_input, cleanup_input = make_temp_copy(source_file)

        with tempfile.NamedTemporaryFile(suffix=".uai", delete=False) as tmp_output:
            output_file = tmp_output.name

        try:
            # Convert UAI to UAI
            exit_code, stdout, stderr = run_convert_pgm(
                [temp_input, "--uai", output_file, "--quiet"]
            )
            assert exit_code == 0

            # Load both models and compare
            pgm_orig = load_model(temp_input)
            pgm_output = load_model(output_file)

            # Check same number of nodes
            assert len(pgm_orig.nodes) == len(pgm_output.nodes)

            # Check same number of CPDs
            assert len(pgm_orig.cpds) == len(pgm_output.cpds)

        finally:
            cleanup_input()
            if os.path.exists(output_file):
                os.remove(output_file)

    @pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
    def test_bif_uai_bif_roundtrip(self):
        """Test BIF -> UAI -> BIF preserves structure"""
        source_file = os.path.join(cwd, "asia.bif")
        temp_input, cleanup_input = make_temp_copy(source_file)

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
                [temp_input, "--uai", uai_file, "--quiet"]
            )
            assert exit_code == 0

            # UAI -> BIF
            exit_code, stdout, stderr = run_convert_pgm(
                [uai_file, "--bif", bif_file, "--quiet"]
            )
            assert exit_code == 0

            # Compare original and final
            pgm_orig = load_model(temp_input)
            pgm_final = load_model(bif_file)

            assert len(pgm_orig.nodes) == len(pgm_final.nodes)

        finally:
            cleanup_input()
            if os.path.exists(uai_file):
                os.remove(uai_file)
            if os.path.exists(bif_file):
                os.remove(bif_file)
