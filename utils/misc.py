
import numpy as np
import cv2
import heapq
import Augmentor


class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file, allow_pickle=True).mean(1).mean(1)

        self.maker = Augmentor.Pipeline()
        self.maker.rotate(0.7, max_left_rotation=10, max_right_rotation=10)
        self.maker.zoom(0.5, min_factor=1.1, max_factor=1.3)
        self.maker.flip_left_right(0.5)
        # self.maker.random_distortion(0.5, 5, 5, 5)
        self.maker.skew(0.4, 0.5)

    def image_distortion(self, image):
        return self.maker._execute_with_array(image)

    def load_image(self, image_file, distortion=False):
        """ Load and preprocess an image. """
        image = cv2.imread(image_file)

        if distortion:
            image = self.image_distortion(image)

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        # image = image - self.mean

        return image

    def load_images(self, image_files, distortion=False):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file, distortion))
        images = np.array(images, np.float32)
        return images


class CaptionData(object):
    def __init__(self, sentence, memory, output, score):
       self.sentence = sentence
       self.memory = memory
       self.output = output
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score


class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []
