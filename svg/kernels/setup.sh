# if nvjitlink not in LD_LIBRARY_PATH, add it
if [[ ":$LD_LIBRARY_PATH:" != *":$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):"* ]]; then
    export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
fi

mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j