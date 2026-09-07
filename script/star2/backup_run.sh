#!/bin/bash
# backup_run.sh — copy one finished run directory into the project's backup store and prove the copy.
#
#   backup_run.sh <src-run-dir> <store-root> [<run-name>]
#
# Result: <store-root>/outputs/<run-name>/ mirrors <src-run-dir>. Every file is sha256-hashed on both
# ends; the destination hashes are written to <dst>/.backup_sha256.txt and one line is appended to
# <store-root>/outputs/.backup_log.tsv. Exit 0 only when every hash matches. Never deletes anything.
#
# Why --inplace: on the jupiter export, curated checkpoints may be hardlinked; rsync's default
# temp-file-then-rename would silently break those links (see the lightgen store design note).
# Why a separate CPU job on star2: copying 100+ GB over NFS takes tens of minutes and must not hold
# GPUs idle — the training script submits this on the same node after it finishes.
set -u
SRC=${1:?usage: backup_run.sh <src-run-dir> <store-root> [<run-name>]}
STORE=${2:?usage: backup_run.sh <src-run-dir> <store-root> [<run-name>]}
RUN=${3:-$(basename "$SRC")}
DST="$STORE/outputs/$RUN"
LOG="$STORE/outputs/.backup_log.tsv"
[ -d "$SRC" ] || { echo "backup_run: source is not a directory: $SRC" >&2; exit 2; }
[ -d "$STORE/outputs" ] || { echo "backup_run: store has no outputs/ directory: $STORE" >&2; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/backup_run.XXXXXX"); trap 'rm -rf "$T"' EXIT
hash_tree() { ( cd "$1" && find . -type f ! -name '.backup_sha256.txt' -print0 | sort -z | xargs -0 -r sha256sum ); }

echo "backup_run: $SRC -> $DST  ($(date -Is), $(hostname -s))"
mkdir -p "$DST"
rsync -a --inplace "$SRC"/ "$DST"/ || { echo "backup_run: rsync failed" >&2; exit 1; }
hash_tree "$SRC" > "$T/src.sha256"
hash_tree "$DST" > "$T/dst.sha256"
n=$(wc -l < "$T/src.sha256"); bytes=$(du -sb "$SRC" | cut -f1)
if diff -q "$T/src.sha256" "$T/dst.sha256" >/dev/null; then status=verified; else status=MISMATCH; fi
cp "$T/dst.sha256" "$DST/.backup_sha256.txt"
( flock 9; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$(hostname -s)" "$RUN" "$n" "$bytes" "$status" "$SRC" >&9 ) 9>>"$LOG"
if [ "$status" = verified ]; then
  echo "backup_run: verified $n files, $bytes bytes -> $DST"; exit 0
else
  echo "backup_run: HASH MISMATCH — differing entries:" >&2; diff "$T/src.sha256" "$T/dst.sha256" | head -20 >&2; exit 1
fi
