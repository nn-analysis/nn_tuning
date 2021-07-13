#!/bin/bash
set -eu

die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in pdoc; do
    command -v "$cmd" >/dev/null ||
        die "Missing $cmd; \`pip install $cmd\`"
done

DOCROOT="$(dirname "$(readlink -f "$0")")"
BUILDROOT="$DOCROOT/build"

echo
echo 'Building API reference docs'
echo
mkdir -p "$BUILDROOT"
rm -r "$BUILDROOT" 2>/dev/null || true
pushd "$DOCROOT/.." >/dev/null

pdoc -d google -o "$BUILDROOT" \
     nn_tuning