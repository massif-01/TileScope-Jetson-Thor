set -euxo pipefail
mkdir -p ~/tilescope/sources
cd ~/tilescope/sources
sha=7fce5439fe451cd48340b112ec37d2befe02c776
curl -fLsS --retry 2 --connect-timeout 10 --max-time 240 https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$sha -o llama.tar.gz
echo '90f1143f04bea3e88c12322314e2ac12e4489bec756ad555b670587fdfe41040  llama.tar.gz' | sha256sum --check -
tar xf llama.tar.gz
src="$PWD/llama.cpp-$sha"
cmake -S "$src" -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server -j4
