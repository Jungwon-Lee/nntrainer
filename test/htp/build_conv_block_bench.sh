#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build the synthetic conv block check against the skel's generated IDL.
set -eu
: "${ANDROID_NDK:?set ANDROID_NDK}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CC="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
CXX="${CC}++"
SDK_INC="$HEXAGON_SDK_ROOT/incs"
FASTRPC="$HEXAGON_SDK_ROOT/ipc/fastrpc"
mkdir -p "$SCRIPT_DIR/build"
"$CC" -O2 -fPIC -I "$SCRIPT_DIR/generated" -I "$SDK_INC" \
  -I "$SDK_INC/stddef" -I "$FASTRPC/incs" \
  -c "$SCRIPT_DIR/generated/nntr_hvx_stub.c" -o "$SCRIPT_DIR/build/conv_stub.o"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -static-libstdc++ \
  -I "$SCRIPT_DIR/generated" -I "$SDK_INC" -I "$SDK_INC/stddef" \
  -I "$FASTRPC/incs" -I "$REPO_ROOT/nntrainer/tensor/htp_backend/hmx" \
  "$SCRIPT_DIR/conv_block_bench.cpp" "$SCRIPT_DIR/build/conv_stub.o" \
  -L "$FASTRPC/remote/ship/android_aarch64" -lcdsprpc -llog -ldl \
  -o "$SCRIPT_DIR/build/conv_block_bench"
