#!/bin/sh
# Assemble blender/dist/nebula_blender(.zip): add-on + ctypes binding + libnebula.so
set -e
cd "$(dirname "$0")/.."
make -s lib
rm -rf blender/dist && mkdir -p blender/dist
cp -r blender/nebula_blender blender/dist/
cp python/nebula.py blender/dist/nebula_blender/nebula.py
cp build/libnebula.so blender/dist/nebula_blender/libnebula.so
(cd blender/dist && zip -qr nebula_blender.zip nebula_blender -x '*/__pycache__/*')
echo "built blender/dist/nebula_blender.zip"
