# Copy to env.sh, set the two paths below, then: source env.sh
# Refuses to continue if the project drive is missing, so nothing falls back to the internal disk.
# Tool managed files live in env.sparsebundle, an APFS image stored in this folder and mounted
# at /Volumes/KWenv. exFAT breaks installers through ._ sidecar files; see docs/storage_audit.md.
# Run scripts/detach_env.sh before ejecting the drive.
export KW_ROOT=/Volumes/YOUR_DRIVE/KnowWhen-R
KW_DRIVE=/Volumes/YOUR_DRIVE
export KW_ENV=/Volumes/KWenv
if ! mount | grep -q "on $KW_DRIVE " || [ ! -d "$KW_ROOT" ]; then
  echo "Project drive is not mounted. Stopping." >&2
  return 1 2>/dev/null || exit 1
fi
if ! mount | grep -q "on $KW_ENV "; then
  hdiutil attach "$KW_ROOT/env.sparsebundle" -nobrowse >/dev/null 2>&1
fi
if ! mount | grep -q "on $KW_ENV "; then
  echo "Could not mount env.sparsebundle. Stopping." >&2
  return 1 2>/dev/null || exit 1
fi
export HF_HOME=$KW_ENV/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_XET_CACHE=$HF_HOME/xet
export KAGGLE_CONFIG_DIR="$KW_ENV/cache/kaggle"
export TORCH_HOME=$KW_ENV/cache/torch
export PIP_CACHE_DIR=$KW_ENV/cache/pip
export UV_CACHE_DIR=$KW_ENV/cache/uv
export UV_PYTHON_INSTALL_DIR=$KW_ENV/cache/uv-python
export UV_TOOL_DIR=$KW_ENV/cache/uv-tools
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=$KW_ENV/venv
export XDG_CACHE_HOME=$KW_ENV/cache
export XDG_DATA_HOME=$KW_ENV/cache/xdg-data
export XDG_STATE_HOME=$KW_ENV/cache/xdg-state
export MPLCONFIGDIR=$KW_ENV/cache/matplotlib
export NLTK_DATA=$KW_ENV/cache/nltk
export PYTHONPYCACHEPREFIX=$KW_ENV/cache/pycache
export TMPDIR=$KW_ENV/tmp
export TEMP=$TMPDIR TMP=$TMPDIR
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export COPYFILE_DISABLE=1
mkdir -p "$TMPDIR" "$HF_HOME" "$MPLCONFIGDIR"
[ -f "$KW_ENV/venv/bin/activate" ] && . "$KW_ENV/venv/bin/activate"
true
