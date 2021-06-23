#!/bin/bash
set -eu

die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in pdoc3; do
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

pdoc3 --html \
     --output-dir "$BUILDROOT" \
     nn_analysis
popd >/dev/null

echo
echo 'Testing for broken links'
echo
pushd "$BUILDROOT" >/dev/null
grep -PR '<a .*?href=' |
    sed -E "s/:.*?<a .*?href=([\"'])(.*?)/\t\2/g" |
    tr "\"'" '#' |
    cut -d'#' -f1 |
    sort -u -t$'\t' -k 2 |
    sort -u |
    python -c '
import sys
from urllib.parse import urljoin
for line in sys.stdin.readlines():
    base, url = line.split("\t")
    print(base, urljoin(base, url.strip()), sep="\t")
    ' |
    grep -v $'\t''$' |
    while read -r line; do
        while IFS=$'\t' read -r file url; do
            [ -f "$url" ] ||
                curl --silent --fail --retry 5 --retry-delay 5 --user-agent 'Mozilla/5.0 Firefox 61' "$url" >/dev/null 2>&1 ||
                die "broken link in $file:  $url"
        done
    done
popd >/dev/null


echo
echo "All good. Docs in: $BUILDROOT"
echo
echo "    file://$BUILDROOT/nn_analysis/index.html"
echo
