#!/usr/bin/python3

import os
import atexit
import io
import sys
import collections
import picamera
import picamera.array
import numpy as np
from scipy import ndimage
import time
import threading
import queue
import signal
import logging
from collections import deque

# a no-op to handle the profile decorator -- or uncomment the profile import below
def profile(func):
   def func_wrapper(*args, **kwargs):
       return func(*args, **kwargs)
   return func_wrapper

#from profilehooks import profile

class MotionVectorReader(picamera.array.PiMotionAnalysis):
	"""This is a hardware-assisted motion detector, able to process a high-definition
	video stream (but not full HD, and at 10 fps only) on a Raspberry Pi 1, despite
	being implemented in Python. How is that possible?! The magic is in computing
	from H.264 motion vector data only: the Pi camera outputs 16x16 macro block MVs,
	so we only have about 5000 blocks per frame to process. Numpy is fast enough for that.
	"""

	area = 0
	frames = 0
	window = 0
	camera = None
	trigger = threading.Event()
	output = None

	def __str__(self):
		return "sensitivity {0}/{1}".format(self.area, self.frames)

	def __init__(self, camera, window=10, area = 25, frames = 4):
		"""Initialize motion vector reader

		Parameters
		----------
		camera : PiCamera
		size : minimum number of connected MV blocks (each 16x16 pixels) to qualify for movement
		frames : minimum number of frames to contain movement to quality
		"""
		super(type(self), self).__init__(camera)
		self.camera = camera
		self.area = area
		self.frames = frames
		self.window = window
		self._last_frames = deque(maxlen=window)
		logging.debug("motion detection sensitivity: "+str(self))

	def save_motion_vectors(self,file):
		self.output = open(file,"ab")

	def set(self):
		self.trigger.set()

	def clear(self):
		self.trigger.clear()

	def motion(self):
		return self.trigger.is_set()

	def wait(self, timeout = 0.0):
		return self.trigger.wait(timeout)

	disabled = False
	_last_frames = deque(maxlen=10)
	noise = None
	@profile
	def analyse(self, a):
		"""Runs once per frame on a 16x16 motion vector block buffer (about 5000 values).
		Must be faster than frame rate (max 100 ms for 10 fps stream).
		Sets self.trigger Event to trigger capture.
		"""

		if self.disabled:
			self._last_frames.append(False)
			return

		import struct
		if self.output:
			self.output.write(struct.pack('>8sL?8sBBB',
				b'frameno\x00', self.camera.frame.index, self.motion(),
				b'mvarray\x00', a.shape[0], a.shape[1], a[0].itemsize))
			self.output.write(a)

		# the motion vector array we get from the camera contains three values per
		# macroblock: the X and Y components of the inter-block motion vector, and
		# sum-of-differences value. the SAD value has a completely different meaning
		# on a per-frame basis, but abstracted over a longer timeframe in a mostly-still
		# video stream, it ends up estimating noise pretty well. Accordingly, we
		# can use it in a decay function to reduce sensitivity to noise on a per-block
		# basis

		# accumulate and decay SAD field
		noise = self.noise
		if not noise:
			noise = np.zeros(a.shape,dtype=np.short)
		shift = max(self.window.bit_length()-2,0)
		noise -= ( noise >> shift ) + 1 # decay old noise
		noise = np.add(noise, a['sad'] >> shift).clip(0)

		# then look for motion vectors exceeding the length of the current mask
		a = np.sqrt(
			np.square(a['x'].astype(np.float)) +
			np.square(a['y'].astype(np.float))
			).clip(0, 255).astype(np.uint8)
		self.field = a

		# look for the largest continuous area in picture that has motion
		mask = (a > (noise >> 4)) # every motion vector exceeding current noise field
		labels,count = ndimage.label(mask) # label all motion areas
		sizes = ndimage.sum(mask, labels, range(count + 1)) # number of MV blocks per area
		largest = np.sort(sizes)[-1] # what's the size of the largest area

		# Do some extra work to clean up the preview overlay. Remove all but the largest
		# motion region, and even that if it's just one MV block (considered noise)
		#mask = (sizes < max(largest,2))
		#mask = mask[labels] # every part of the image except for the largest object
		#self.field = mask

		# TODO: all the regions (and small movement) that we discarded as non-essential:
		# should feed that to a subroutine that weights that kind of movement out of the
		# picture in the future for auto-adaptive motion detector

		# does that area size exceed the minimum motion threshold?
		motion = (largest >= self.area)
		# then consider motion repetition
		self._last_frames.append(motion)

		def count_longest(a,value):
			ret = i = 0
			while i < len(a):
				for j in range(0,len(a)-i):
					if a[i+j] != value: break
					ret = max(ret,j+1)
				i += j+1
			return ret
		longest_motion_sequence = count_longest(self._last_frames, True)

		if longest_motion_sequence >= self.frames:
			self.set()
		elif longest_motion_sequence < 1:
			# clear motion flag once motion has ceased entirely
			self.clear()
		return motion

class MotionRecorder(threading.Thread):
	"""Record video into a circular memory buffer and extract motion vectors for
	simple motion detection analysis. Enables dumping the video frames out to file
	if motion is detected.
	"""

	# half of hardware resolution leaves us HD 4:3 and provides 2x2 binning
	# for V1 camera: 1296x972, for V2 camera: 1640x1232. Also use sensor_mode: 4
	width = 1296
	height = 972
	framerate = 10 # lower framerate for more time on per-frame analysis
	bitrate = 2000000 # 2Mbps is a high quality stream for 10 fps HD video
	prebuffer = 10 # number of seconds to keep in buffer
	postbuffer = 5 # number of seconds to record post end of motion
	overlay = False
	file_pattern = '%y-%m-%dT%H-%M-%S.h264' # filename pattern for time.strfime
	_area = 25 # number of connected MV blocks (each 16x16 pixels) to count as a moving object
	_frames = 4 # number of frames which must contain movement to trigger

	_camera = None
	_motion = None
	_output = None

	def __enter__(self):
		self.start_camera()
		threading.Thread(name="blink", target=self.blink, daemon=True).start()
		threading.Thread(name="annotate", target=self.annotate_with_datetime, args=(self._camera,), daemon=True).start()
		if self.overlay: threading.Thread(name="motion overlay", target=self.motion_overlay, daemon=True).start()
		logging.info("now ready to detect motion")
		return self

	def __exit__(self,type,value,traceback):
		camera = self._camera
		if camera.recording:
			camera.stop_recording()

	def __init__(self, overlay=False):
		super().__init__()
		self.overlay = overlay

	def wait(self,timeout = 0.0):
		"""Use this instead of time.sleep() from sub-threads so that they would
		wake up to exit quickly when instance is being shut down.
		"""
		try:
			self._camera.wait_recording(timeout)
		except picamera.exc.PiCameraNotRecording:
			# that's fine, return immediately
			pass

	@property
	def area(self):
		return self._area
	@area.setter
	def area(self,value):
		self._area=value
		if self._motion: self._motion.area=value
	@property
	def frames(self):
		return self._frames
	@frames.setter
	def frames(self,value):
		self._frames=value
		if self._motion: self._motion.frames=value

	def start_camera(self):
		"""Sets up PiCamera to record H.264 High/4.1 profile video with enough
		intra frames that there is at least one in the in-memory circular buffer when
		motion is detected."""
		self._camera = camera = picamera.PiCamera(clock_mode='raw', sensor_mode=4,
			resolution=(self.width, self.height), framerate=self.framerate)
		#camera.sensor_mode = 4
		if self.overlay: camera.start_preview(alpha=255)
		self._stream = stream = picamera.PiCameraCircularIO(camera,
			seconds=self.prebuffer+1, bitrate=self.bitrate)
		self._motion = motion = MotionVectorReader(camera, window=self.postbuffer*self.framerate,
			area=self.area, frames=self.frames)
		camera.start_recording(stream, motion_output=motion,
			format='h264', profile='high', level='4.1', bitrate=self.bitrate,
			inline_headers=True, intra_period=self.prebuffer*self.framerate // 2)
		camera.wait_recording(1) # give camera some time to start up

	captures = queue.Queue()

	def capture_jpeg(self):
		camera = self._camera
		camera.capture('/dev/shm/picamera.jpg', use_video_port=True, format='jpeg', quality=80)

	def run(self):
		"""Main loop of the motion recorder. Waits for trigger from the motion detector
		async task and writes in-memory circular buffer to file every time it happens,
		until motion detection trigger. After each recording, the name of the file
		is posted to captures queue, where whatever is consuming the recordings can
		pick it up.
		"""
		self._motion.disabled = False
		while self._camera.recording:
			# wait for motion detection
			if self._motion.wait(self.prebuffer):
				if self._motion.motion():
					self._camera.led = True
					try:
						# start a new video, then append circular buffer to it until
						# motion ends
						name = time.strftime(self.file_pattern)
						output = io.open(name, 'wb')
						self.append_buffer(output,header=True)
						while self._motion.motion() and self._camera.recording:
							self.wait(self.prebuffer / 2)
							self.append_buffer(output)
					except picamera.PiCameraError as e:
						logging.error("while saving recording: "+e)
						pass
					finally:
						output.close()
						# also store a JPEG of the current view
						self.capture_jpeg()
						self._output = None
						self._camera.led = False
						self.captures.put(name)
					# wait for the circular buffer to fill up before looping again
					self.wait(self.prebuffer / 2)

	def append_buffer(self,output,header=False):
		"""Flush contents of circular framebuffer to current on-disk recording.
		"""
		if header:
			header=picamera.PiVideoFrameType.sps_header
		else:
			header=None
		stream = self._stream
		with stream.lock:
			stream.copy_to(output, seconds=self.prebuffer, first_frame=header)
			#firstframe = lastframe = next(iter(stream.frames)).index
			#for frame in stream.frames: lastframe = frame.index
			#logging.debug("write {0} to {1}".format(firstframe,lastframe))
			stream.clear()
		return output

	def blink(self):
		"""Background thread for blinking the camera LED (to signal detection).
		"""
		while self._camera.recording:
			if not self._motion.motion() and self._output is None:
				self._camera.led = True
				self.wait(0.05) # this is enough for a quick blink
				self._camera.led = False
			self.wait(2-time.time()%2) # wait up to two seconds

	def annotate_with_datetime(self,camera):
		"""Background thread for annotating date and time to video.
		"""
		while camera.recording:
			camera.annotate_text = time.strftime("%y-%m-%d %H:%M") + " " + str(self._motion)
			camera.annotate_background = True
			self.wait(60-time.gmtime().tm_sec) # wait to beginning of minute

	def motion_overlay(self):
		"""Background thread for drawing motion detection mask to on-screen preview.
		Basically for debug purposes.
		"""
		width = (self.width//16//32+1)*32 # MV blocks rounded up to next-32 pixels
		height = (self.height//16//16+1)*16 # MV blocks rounded up to next-16 pixels
		buffer = np.zeros((height,width,3), dtype=np.uint8)
		logging.debug("creating a motion overlay of size {0}x{1}".format(width, height))
		overlay = self._camera.add_overlay(memoryview(buffer),size=(width,height),alpha=128)
		# this thread will exit immediately if motion overlay is configured off
		while self._camera.recording:
			a = self._motion.field # last processed MV frame
			if a is not None:
				# center MV array on output buffer
				w = a.shape[1]
				x = (width-w)//2+1
				h = a.shape[0]
				y = (height-h)//2+1
				# highlight those blocks which exceed thresholds on green channel
				buffer[y:y+h,x:x+w,1] = a * 255
			try:
				overlay.update(memoryview(buffer))
			except picamera.exc.PiCameraRuntimeError as e:
				# it's possible to get a "failed to get a buffer from the pool" error here
				pass
			self.wait(0.5) # limit the preview framerate to max 2 fps
		self._camera.remove_overlay(overlay)

def capture():
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
	try:
		with MotionRecorder() as mr:
			mr.start()
			while True:
				recording = mr.captures.get()
				logging.info("motion capture in '{0}'".format(recording))
				mr.captures.task_done()
	except (KeyboardInterrupt, SystemExit):
		exit()

if __name__ == '__main__':
	capture()
