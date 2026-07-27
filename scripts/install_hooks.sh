#!/usr/bin/env bash
# Install the versioned hook into the repository's common git directory.
# The copied hook remains available when pushing from an older worktree whose
# checkout does not contain scripts/hooks/pre-push.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ROOT="$(cd "$ROOT" && pwd -P)"
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
HOOK_DIR="$COMMON_DIR/stiffgipc-hooks"
SOURCE="$ROOT/scripts/hooks/pre-push"
TARGET="$HOOK_DIR/pre-push"

[ -f "$SOURCE" ] || { echo "missing $SOURCE" >&2; exit 2; }
mkdir -p "$HOOK_DIR"
install -m 0755 "$SOURCE" "$TARGET"
git config --local core.hooksPath "$HOOK_DIR"

echo "installed: $TARGET"
echo "core.hooksPath=$(git config --get core.hooksPath)"
echo "hook_sha256=$(sha256sum "$TARGET" | awk '{print $1}')"
