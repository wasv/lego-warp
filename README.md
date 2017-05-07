# LEGO Warp

Detects a green lego backplane, then crops and warps the image to isolate the backplane.

Used primarily with my LEGO schedule system.

## Building
```bash
    mkdir build
    cd build
    cmake ..
    make
```

## Usage
```bash
    ./lego-detect [path to images] # Detects green lego backplanes. Outputs images as result-[000-999].jpg
    ./sched-read result-*.jpg # Reads resulting images and detects color of brick in each spot. Hardcoded to 64x32 resolution for now.
```
