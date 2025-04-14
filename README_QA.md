# Refer to https://pachterlab.github.io/kallisto/local_build.html
wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7.tar.gz
tar -xf cmake-3.31.7.tar.gz
cd cmake-3.31.7

./bootstrap --prefix=<the abesolute path>/cmake-3.31
make
make install