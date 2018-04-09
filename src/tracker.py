from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
from model import Classifier


class Video:
    def __init__(self, video_file):
        self.video = cv2.VideoCapture(video_file)

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame = 0
        if not(self.width and self.height):
            ret, frame = self.video.read()
            if ret:
                self.height, self.width = frame.shape
                self.current_frame += 1
            else:
                raise ValueError('video empty')

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.duration = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps)

    @property
    def current_time(self):
        try:
            return self.current_frame // self.fps
        except (TypeError, ZeroDivisionError):
            raise AttributeError('fps not defined by the video')

    def forward(self, num_frames):
        count = 0
        while count < num_frames:
            self.video.read()
            count += 1

            self.current_frame += 1

    def read(self):
        ret, frame = self.video.read()
        self.current_frame += 1
        return frame if ret else None

    def close(self):
        self.video.release()


class Tracker:
    def __init__(self, video_file, classifier):
        self.video = Video(video_file)

        self.log = {}
        self.current_frame = None

        self.classifier = classifier

    def skip(self, num_frames):
        self.video.forward(num_frames)

    def track(self, target_class=None, tag_frame=True):
        pass
