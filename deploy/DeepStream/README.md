# Byte-Track Integration with Deepstream
Integrating Byte-Track C++ code with the Deepstream-5.1

## Build Instructions
```
$mkdir build && cd build  

$cmake ..  

$make ByteTracker  
```

This will create lib/libByteTracker.so file which can be passed as the custom low level tracker library to deepstream.

## References
1. [How to Implement a Custom Low-Level Tracker Library in Deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#how-to-implement-a-custom-low-level-tracker-library)
2. [Byte-Track](https://github.com/ifzhang/ByteTrack)
