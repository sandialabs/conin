#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(dirname "$script_dir")

python -m pip install -e "$project_root[docs]"
python -m sphinx -b html "$script_dir" "$script_dir/_build/html"
