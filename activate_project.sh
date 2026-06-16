#!/usr/bin/env bash

# Source this script to activate the local venv and include project root in PYTHONPATH.
# Usage: source ./activate_project.sh

_tardisproject_activate_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_tardisproject_venv_activate="${_tardisproject_activate_script_dir}/venv/bin/activate"

if [[ ! -f "${_tardisproject_venv_activate}" ]]; then
    echo "activate_project.sh: missing venv activation script at ${_tardisproject_venv_activate}" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${_tardisproject_venv_activate}"

case ":${PYTHONPATH:-}:" in
    *":${_tardisproject_activate_script_dir}:"*)
        ;;
    *)
        export PYTHONPATH="${_tardisproject_activate_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
        ;;
esac

# Optional: set USE_PROJECT_IPYTHONDIR=1 before sourcing to use project-local
# IPython config in .ipython/. By default user-level IPython config is used.
if [[ "${USE_PROJECT_IPYTHONDIR:-}" == "1" ]]; then
    export IPYTHONDIR="${_tardisproject_activate_script_dir}/.ipython"
elif [[ "${IPYTHONDIR:-}" == "${_tardisproject_activate_script_dir}/.ipython" ]]; then
    unset IPYTHONDIR
fi

unset _tardisproject_activate_script_dir
unset _tardisproject_venv_activate
