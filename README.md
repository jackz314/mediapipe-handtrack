# Handtracking water ripples

## Summary

This is a simple demo that shows water ripple effects based on user hand interactions.

It uses Google's [MediaPipe](https://mediapipe.dev) ML pipeline and model for hand tracking, [OpenCV](https://opencv.org/) for processing and displaying stuff, and an "ancient" but fast and efficient [algorithm](https://web.archive.org/web/20160505235423/http://freespace.virgin.net/hugo.elias/graphics/x_water.htm) for its water ripple effects.

## Installation

The core demo part can be found in the [hand-paint](hand-paint) folder, the code alone is simple but its dependencies on [MediaPipe](https://mediapipe.dev) is pretty complicated. All of its dependencies can be installed by following the [installation instructions](mediapipe/docs/install.md) ([up-to-date version](https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md)), additionally, the [documentation](https://mediapipe.readthedocs.io) page for [MediaPipe](https://mediapipe.dev) should be helpful if problems occur.

Roughly the dependencies should include [bazel](https://bazel.build), [OpenCV](https://opencv.org/), [FFmpeg](https://www.ffmpeg.org/), and [Mesa](https://www.mesa3d.org/). 

Note that this demo uses the GPU version of the MediaPipe ML model for faster performance, so you need to follow the corresponding GPU installation steps when prompted. This also means that this demo can only be ran on Linux.

Finally, after setting up all the dependencies, to run the demo, just do `make clean-run` in the [hand-paint](hand-paint) folder, for more `make` commands, look into the [Makefile](hand-paint/Makefile).

## Demo

Here's a GIF of the demo:

![demo](hand-paint/demo/demo.gif)

## Other stuff

The default branch for this demo is `jack`. The `master` branch contains [MediaPipe](https://mediapipe.dev)'s code as I forked it for reference.
