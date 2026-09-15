#!/usr/bin/env python3
"""
Command-line tool to convert graphical model files between different formats.

Supported formats:
- UAI format (.uai)
- BIF format (.bif)
- Conin native format
"""

import argparse
import sys
import os

from conin.common.unified import load_model, save_model
from conin.bayesian_network.model import DiscreteBayesianNetwork
from conin.markov_network.model import DiscreteMarkovNetwork
import conin.common.conin


def main():
    parser = argparse.ArgumentParser(
        description="Convert graphical model files between different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert BIF to UAI format with explicit output filename
  %(prog)s input.bif --uai output.uai

  # Convert UAI to BIF format with auto-generated filename (input.bif)
  %(prog)s input.uai --bif

  # Convert to both formats with auto-generated filenames
  %(prog)s model.uai --uai --bif

  # Convert with explicit output filename
  %(prog)s model.bif --uai model.uai
        """,
    )

    parser.add_argument(
        "input_file",
        help="Input graphical model file (supports .uai, .uai.gz, .bif, .bif.gz formats)",
    )

    parser.add_argument(
        "--uai",
        nargs="?",
        const="",
        metavar="OUTPUT_FILE",
        help="Output file in UAI format (.uai). If no filename is specified, generates one from the input filename.",
    )

    parser.add_argument(
        "--bif",
        nargs="?",
        const="",
        metavar="OUTPUT_FILE",
        help="Output file in BIF format (.bif). If no filename is specified, generates one from the input filename.",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational messages",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Check that at least one output format is specified
    if args.uai is None and args.bif is None:
        print(
            "Error: At least one output format must be specified (--uai or --bif)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Helper function to generate output filename
    def generate_output_filename(input_file, new_extension):
        """Generate output filename by replacing input extension with new extension"""
        base = os.path.splitext(input_file)[0]
        # Handle compressed files (.uai.gz, .bif.gz)
        if base.endswith(".uai") or base.endswith(".bif"):
            base = os.path.splitext(base)[0]
        return base + new_extension

    try:
        # Load the input model
        if not args.quiet:
            print(f"Loading model from: {args.input_file}")

        pgm = load_model(args.input_file, model_type="conin", quiet=args.quiet)

        # Save to UAI format if requested
        if args.uai is not None:
            # Generate filename if not provided
            if args.uai == "":
                output_file = generate_output_filename(args.input_file, ".uai")
            else:
                output_file = args.uai if args.uai.endswith(".uai") else args.uai + ".uai"

            if not args.quiet:
                print(f"Saving model to UAI format: {output_file}")

            save_model(pgm, output_file, quiet=args.quiet)

            if not args.quiet:
                print(f"Successfully saved to: {output_file}")

        # Save to BIF format if requested
        if args.bif is not None:
            # Generate filename if not provided
            if args.bif == "":
                output_file = generate_output_filename(args.input_file, ".bif")
            else:
                output_file = args.bif if args.bif.endswith(".bif") else args.bif + ".bif"

            if not args.quiet:
                print(f"Saving model to BIF format: {output_file}")

            # BIF format only supports Bayesian networks, not Markov networks
            if isinstance(pgm, DiscreteMarkovNetwork):
                raise ValueError(
                    "Cannot save Markov network to BIF format. "
                    "BIF format only supports Bayesian networks. "
                    "Use --uai to save in UAI format instead."
                )

            # Convert conin model to pgmpy before saving
            if isinstance(pgm, DiscreteBayesianNetwork):
                pgm_pgmpy = conin.common.conin.convert_conin_to_pgmpy_bn(pgm)
            else:
                raise ValueError(
                    f"Cannot save model of type {type(pgm).__name__} to BIF format. "
                    "Only Bayesian networks are supported."
                )

            save_model(pgm_pgmpy, output_file, model_type="pgmpy", quiet=args.quiet)

            if not args.quiet:
                print(f"Successfully saved to: {output_file}")

    except ImportError as e:
        print(f"Error: Missing required dependency - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Conversion completed successfully")


if __name__ == "__main__":
    main()
