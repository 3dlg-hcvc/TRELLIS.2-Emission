#!/bin/bash
# bash script/internal/gate/compare.sh <dirA> <dirB> [<shas_file>]
# Byte comparison of two gate runs' pred_voxels/*.npz: the shapes in <shas_file> (each must exist on
# both sides, so a subset run compares against a full baseline), or else every file on either side.
# Prints each differing (DIFF) and one-sided (MISSING) file, then "same=N different=D missing=M";
# exits 0 only when D=0, M=0 and N>0.
set -uo pipefail
A="${1:?dirA}/pred_voxels"; B="${2:?dirB}/pred_voxels"
if [ -n "${3:-}" ]; then
  NAMES=$(grep -v '^[[:space:]]*$' "$3" | sed 's/$/.npz/') || { echo "cannot read $3" >&2; exit 2; }
else
  LA=$(ls -U "$A") || { echo "cannot list $A" >&2; exit 2; }
  LB=$(ls -U "$B") || { echo "cannot list $B" >&2; exit 2; }
  NAMES=$(printf '%s\n%s\n' "$LA" "$LB" | grep '\.npz$')
fi
same=0; diff=0; miss=0
while read -r n; do
  [ -n "$n" ] || continue
  if [ -f "$A/$n" ] && [ -f "$B/$n" ]; then
    if cmp -s "$A/$n" "$B/$n"; then same=$((same+1)); else diff=$((diff+1)); echo "DIFF $n"; fi
  else miss=$((miss+1)); echo "MISSING $n"; fi
done < <(printf '%s\n' "$NAMES" | sort -u)
echo "same=$same different=$diff missing=$miss"
[ "$diff" = 0 ] && [ "$miss" = 0 ] && [ "$same" -gt 0 ]
