#!/bin/bash

# Usage: bash svg_to_png.sh {in_dir} {out_dir} {width}
# in_dir: Input directory with *.svg files
# out_dir: Output directory (writing *.png files to)
# width: Desired width of output *.png images

mkdir -p "$2"

for file_name in "$1"/*.svg; do
    base_name=$(basename "$file_name" .svg)
    inkscape --export-width="$3" --export-png="$2"/"$base_name".png "$1"/"$base_name".svg
done
