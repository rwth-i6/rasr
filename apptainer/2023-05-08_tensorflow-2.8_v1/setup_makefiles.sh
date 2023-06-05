#!/bin/bash
SCRIPT_DIR=$(dirname -- "$0")
BASE_DIR=$SCRIPT_DIR/../..
MAKEFILES_DIR=$SCRIPT_DIR/makefiles

files=("Makefile" "Config.make" "Modules.make" "Options.make" "Rules.make" "config/cc-gcc.make" "config/os-linux.make" "config/proc-aarch64.make" "config/proc-x86_64.make")

for file in "${files[@]}"; do
    cp $MAKEFILES_DIR/$file $BASE_DIR/$file
done
