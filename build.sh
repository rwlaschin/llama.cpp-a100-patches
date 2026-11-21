#!/bin/sh
# Build llama-server from llama.cpp 69ee2ecc with this patch stack.
# Usage: ./build.sh [dest-dir] [cuda-archs]   (defaults: ./llama.cpp, "80;89")
set -e
DEST=${1:-llama.cpp}
ARCHS=${2:-80;89}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=69ee2ecc0150ba7d1204a7f7393dfd651569db8c
PATCHES="gdn sampler yonce skeff rsview st jcap pad3 it2 it3a it56s it7a cmake l4gate i64 topk1"

[ -d "$DEST" ] || git clone https://github.com/ggml-org/llama.cpp "$DEST"
git -C "$DEST" checkout -q "$COMMIT"
for p in $PATCHES; do
  echo "applying $p.patch"
  git -C "$DEST" apply -p1 "$HERE/patches/$p.patch"
done

cmake -S "$DEST" -B "$DEST/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA_MMQ_Y_ONCE=ON -DGGML_CUDA_MMQ_STAGE=ON -DGGML_CUDA_MMQ_I2F_EXACT=ON \
  -DGGML_CUDA_MMQ_Q8_CACHE=ON -DGGML_CUDA_MMQ_Y_ASYNC=ON -DGGML_CUDA_MMQ_CUBLAS_ABOVE=512 \
  -DGGML_CUDA_MMQ_I64=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
cmake --build "$DEST/build" --config Release -j --target llama-server
echo "built $DEST/build/bin/llama-server"
