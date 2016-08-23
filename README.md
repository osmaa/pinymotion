# pinymotion

Python implementation of a motion detecting H.264 camera for Raspberry Pi

Leveraging the ability of the Raspberry Pi camera to produce hardware-encoded
H.264 video and the MMAL API's built-in facility for providing the H.264 motion vectors
as a side-band stream, this Python script is able to detect moving objects in
the field of view with plenty of CPU to spare -- it runs at about 25% CPU usage on the
original single-core 700MHz Pi Model B!

The core of the motion detector works on 16x16 pixel motion vectors, within which it
looks for larger areas of significant motion. By default, an area of at least 25
contiguous MV blocks, each with a vector of at least 10 units (in any direction) is
required for motion to be considered. Finally, the detector expects to see 4 or more
consecutive frames with movement. Adjusting these three parameters will impact the
sensitivity of the recorder.

All hard work is performed by PiCamera http://picamera.readthedocs.io/en/release-1.12/
and Numpy/Scipy http://docs.scipy.org/doc/.

In principle, the same approach could be used with USB and network cameras producing
H.264 streams. Those do not provide the handy motion vector side channel, though, so
a H.264 parser and motion vector extractor will be required.

