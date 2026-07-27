#!/usr/bin/env bash
# Fast, GPU-free contract test for the exact-SHA pre-push dispatcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
HOOK="$ROOT/scripts/hooks/pre-push"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/stiffgipc-hook-test.XXXXXX")"

cleanup()
{
    case "$TEST_ROOT" in
        "${TMPDIR:-/tmp}"/stiffgipc-hook-test.*)
            rm -rf -- "$TEST_ROOT"
            ;;
        *)
            echo "[hook-test] refusing to clean unexpected path: $TEST_ROOT" >&2
            ;;
    esac
}
trap cleanup EXIT

REPO="$TEST_ROOT/repo"
mkdir -p "$REPO"
git -C "$REPO" init -q
git -C "$REPO" config user.name "StiffGIPC Gate Test"
git -C "$REPO" config user.email "gate-test@example.invalid"

mkdir -p "$REPO/scripts"
printf '2\n' > "$REPO/scripts/GATE_PROTOCOL_VERSION"
printf 'A\n' > "$REPO/fixture.txt"
cat > "$REPO/scripts/run_push_gates.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
tier=${1:?tier}
[ "$(cat fixture.txt)" = "A" ] || {
    echo "validator used caller checkout instead of outgoing ref-target commit" >&2
    exit 41
}
case "$tier" in
    full|heavy) ;;
    *) exit 42 ;;
esac
printf 'stub_sha=%s tier=%s\n' "$(git rev-parse HEAD)" "$tier"
STUB
chmod +x "$REPO/scripts/run_push_gates.sh"
git -C "$REPO" add .
git -C "$REPO" commit -qm "fixture A"
SHA_A="$(git -C "$REPO" rev-parse HEAD)"

# Move the caller checkout to a deliberately incompatible tree. A correct hook
# still checks SHA_A in its own detached validator worktree.
printf 'B\n' > "$REPO/fixture.txt"
git -C "$REPO" add fixture.txt
git -C "$REPO" commit -qm "fixture B"
SHA_B="$(git -C "$REPO" rev-parse HEAD)"

run_hook()
{
    local input=$1
    printf '%s' "$input" | (cd "$REPO" && "$HOOK" test-remote unused-url)
}

ZERO=0000000000000000000000000000000000000000
run_hook "refs/heads/a $SHA_A refs/heads/a $ZERO
"
grep -q "stub_sha=$SHA_A tier=full" "$REPO/.git/gate-logs/"*/gates.log

run_hook "refs/tags/v1 $SHA_A refs/tags/v1 $ZERO
"
grep -q "stub_sha=$SHA_A tier=heavy" "$REPO/.git/gate-logs/"*/gates.log

# Multiple refs to one commit are de-duplicated, and a tag upgrades the single
# validation from full to heavy.
before="$(wc -l < "$REPO/.git/gate-logs/history.log")"
run_hook "refs/heads/a $SHA_A refs/heads/a $ZERO
refs/tags/v2 $SHA_A refs/tags/v2 $ZERO
"
after="$(wc -l < "$REPO/.git/gate-logs/history.log")"
[ "$after" -eq $((before + 1)) ]
tail -1 "$REPO/.git/gate-logs/history.log" | grep -q " PASS heavy $SHA_A "

# Emergency bypass is auditable and does not invoke the incompatible SHA_B
# runner from the caller checkout.
printf 'refs/heads/b %s refs/heads/b %s\n' "$SHA_B" "$ZERO" |
    (cd "$REPO" && SKIP_GATES=1 "$HOOK" test-remote unused-url)
grep -q "refs/heads/b@${SHA_B:0:12}->refs/heads/b" \
    "$REPO/.git/gate-logs/bypass.log"

# An old/out-of-contract commit cannot silently run a different gate protocol.
git -C "$REPO" rm -q scripts/GATE_PROTOCOL_VERSION
git -C "$REPO" commit -qm "remove protocol"
SHA_OLD="$(git -C "$REPO" rev-parse HEAD)"
if run_hook "refs/heads/old $SHA_OLD refs/heads/old $ZERO
"; then
    echo "[hook-test] missing protocol unexpectedly passed" >&2
    exit 1
fi

# Ref deletion has no outgoing tree and is a valid no-op.
run_hook "refs/heads/delete $ZERO refs/heads/delete $SHA_A
"

echo "PRE-PUSH-HOOK-TEST: PASS"
