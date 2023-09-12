Image based on official nvidia image tensorflow:22.04-tf2-py3.
 - For release notes see https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_22-04.html

Contains:
 - Ubuntu 20.04
 - CUDA 11.6
 - Tensorflow 2.8
 - Python 3.8
 - i6 Cache Manager
 - Basics required to compile RASR

Issues:
 - Ffmpeg module needs to be turned off for now since API has changed with never versions
 - OpenFST is not installed and related modules are disabled
 - The definition file specifies a tensorflow build for compute capabilities 6.1, 7.5 and 8.6;
 for others, adapt the build command in the file before creating the image.

