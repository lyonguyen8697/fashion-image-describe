
import numpy as np
import cv2
import heapq
import Augmentor
import os
import matplotlib.pyplot as plt
from textwrap import wrap
from googletrans import Translator


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


class ImageSaver(object):
    def __init__(self):

        self.translator = Translator()
        self.translate_language = 'vi'

    def save_test_image(self, image_file, caption, save_dir, translate_cap=False):
        image_name = os.path.basename(image_file)
        image_name = image_name.split('.')[0]

        image = plt.imread(image_file)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].axis('off')

        splitted_caption = wrap(caption, width=40)
        splitted_caption.insert(0, 'Description:')
        row_height = 1.0
        for line in splitted_caption:
            row_height = row_height - 0.05
            axes[1].text(0, row_height, line)

        if translate_cap:
            try:
                translated = self.translator.translate(caption, dest=self.translate_language).text
                translated = wrap(translated, width=40)
                translated.insert(0, 'Translation:')
                row_height = row_height - 0.05
                for line in translated:
                    row_height = row_height - 0.05
                    axes[1].text(0, row_height, line)
            except:
                pass


        fig.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 image_name + '_result.jpg'))

    def save_eval_image(self, image_file, ground_truth_cap, predict_cap, save_dir):
        image_name = os.path.basename(image_file)
        image_name = image_name.split('.')[0]

        image = plt.imread(image_file)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].axis('off')

        ground_truth_cap = wrap(ground_truth_cap, width=40)
        ground_truth_cap.insert(0, 'Ground truth:')
        row_height = 1.0
        for line in ground_truth_cap:
            row_height = row_height - 0.05
            axes[1].text(0, row_height, line)

        predict_cap = wrap(predict_cap, width=40)
        predict_cap.insert(0, 'Prediction:')
        row_height = row_height - 0.05
        for line in predict_cap:
            row_height = row_height - 0.05
            axes[1].text(0, row_height, line)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 image_name + '_result.jpg'))


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

