#!/bin/bash
SCRIPT_DIR=$(dirname -- "$0")
BASE_DIR=$SCRIPT_DIR/../..
MAKEFILES_DIR=$SCRIPT_DIR/makefiles

files=("Makefile" "Makefile.cfg" "Config.make" "Modules.make" "Options.make" "Rules.make" "config/cc-clang.make" "config/cc-gcc.make" "config/cc-icc.make" "config/os-linux.make" "config/os-darwin.make" "config/proc-aarch64.make" "config/proc-x86_64.make")

for file in "${files[@]}"; do
    echo "Copy $file..."
    cp $MAKEFILES_DIR/$file $BASE_DIR/$file
done

echo "Finished setup."
