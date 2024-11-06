Image based on official nvidia image nvcr.io/nvidia/pytorch:23.12-py3.
 - For release notes see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-12.html

Contains:
 - Ubuntu 22.04
 - CUDA 12.3.2
 - Pytorch 2.2
 - ONNX 1.16.3
 - Python 3.10
 - i6 Cache Manager
 - Basics required to compile RASR

Issues:
 - Ffmpeg module needs to be turned off for now since API has changed with never versions
 - OpenFST is not installed and related modules are disabled
 - No tensorflow

