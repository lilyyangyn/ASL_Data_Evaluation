# ASL

Build:

```bash
mkdir build
cd build
cmake ..
make
```

Release build (no debug output):

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Count flops:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCOUNT_FLOPS=yes
make
```

Custom Releae flags:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -O3 -ffast-math"
```