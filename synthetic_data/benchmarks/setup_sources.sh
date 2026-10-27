#!/usr/bin/env bash
# Clone every upstream benchmark into _src/ at the commit the runners were tested with, then apply the
# compatibility patches in patches/. Idempotent: an existing clone is checked for the pinned commit and left alone.
# Usage (from synthetic_data/benchmarks/, Git Bash):  bash setup_sources.sh      # ~2 min, ~620 MB (realcause is 604 MB)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p _src

# name|repo|commit
SOURCES="
xai-bench|abacusai/xai-bench|f0431a7c1628a8b66fd322ce86ddcba0d81201d5
OpenXAI|AI4LIFE-GROUP/OpenXAI|a18288620464250856b55234266a6d1dabb64656
ITErpretability|vanderschaarlab/ITErpretability|8121b9f78c766e0a4a41c4c03e6378a6b8f5c20d
realcause|bradyneal/realcause|7411cc26f10d750aa011b25f2a46cb572d8b8540
ds-wgan|gsbDBI/ds-wgan|ee7fa679dfda852cdb04cef211f2370599e657b6
credence-to-causal-estimation|amzn/credence-to-causal-estimation|195cbf6febf13d53dcb1a864b80fd39794dfe584
fb2ts|miquelmn/fb2ts|d49d3a7905883ec6e2861a46488ad0b28cf6619f
regmode-code|ehsanx/regmode-code|ba593e483c3d95d9f15533c50060b916dd847bd0
"

for line in $SOURCES; do
  IFS='|' read -r name repo commit <<< "$line"
  dest="_src/$name"
  if [ -d "$dest/.git" ]; then
    have=$(git -C "$dest" rev-parse HEAD)
    if [ "$have" != "$commit" ]; then
      echo "ERROR: $dest is at $have, expected $commit" >&2
      exit 1
    fi
    echo "ok   $name ($commit)"
  else
    git init -q "$dest"
    git -C "$dest" remote add origin "https://github.com/$repo.git"
    git -C "$dest" fetch -q --depth 1 origin "$commit"
    git -C "$dest" checkout -q FETCH_HEAD
    echo "new  $name ($commit)"
  fi
  patch="patches/$name.patch"
  if [ -f "$patch" ]; then
    if git -C "$dest" apply --check "../../$patch" 2>/dev/null; then
      git -C "$dest" apply "../../$patch"
      echo "     applied $patch"
    elif git -C "$dest" apply --reverse --check "../../$patch" 2>/dev/null; then
      echo "     $patch already applied"
    else
      echo "ERROR: $patch neither applies nor is already applied in $dest" >&2
      exit 1
    fi
  fi
done
