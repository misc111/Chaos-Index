#!/bin/zsh
set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"
cd "$script_dir"

# Finder-launched `.command` files often inherit a minimal PATH that resolves to
# the system/Xcode Python instead of the interpreter with this repo's deps.
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:${PATH}"

python_bin=""
python_candidates=(
  "$script_dir/.venv/bin/python"
  "$script_dir/.venv/bin/python3"
  "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
  "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
  "/opt/homebrew/bin/python3"
  "/usr/local/bin/python3"
)

for candidate in "${python_candidates[@]}"; do
  if [[ -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done

if [[ -z "$python_bin" ]]; then
  python_bin="$(command -v python3 || true)"
fi

if [[ -z "$python_bin" ]]; then
  echo "Unable to find a usable python3 interpreter for hard refresh." >&2
  exit 1
fi

exec make hard_refresh PYTHON="$python_bin" "$@"
