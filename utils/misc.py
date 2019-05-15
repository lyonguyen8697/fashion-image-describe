
import numpy as np
import cv2
import heapq
import Augmentor
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from colorsys import hsv_to_rgb
from textwrap import wrap
from googletrans import Translator
import string


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
    def __init__(self, image_shape, end_token):

        self.translator = Translator()
        self.translate_language = 'vi'
        self.image_shape = image_shape
        self.end_token = end_token

    def save_test_image(self, image_file, caption, save_dir, translate_cap=False):
        image_name = os.path.basename(image_file)
        image_name = image_name.split('.')[0]

        image = plt.imread(image_file)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].axis('off')

        caption = self._format_caption(caption)
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

        ground_truth_cap = self._format_caption(ground_truth_cap)
        ground_truth_cap = wrap(ground_truth_cap, width=40)
        ground_truth_cap.insert(0, 'Ground truth:')
        row_height = 1.0
        for line in ground_truth_cap:
            row_height = row_height - 0.05
            axes[1].text(0, row_height, line)

        predict_cap = self._format_caption(predict_cap)
        predict_cap = wrap(predict_cap, width=40)
        predict_cap.insert(0, 'Prediction:')
        row_height = row_height - 0.05
        for line in predict_cap:
            row_height = row_height - 0.05
            axes[1].text(0, row_height, line)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 image_name + '_result.jpg'))

    def save_visualization_image(self, image_file, caption, depth_attention_weight, soft_attention_weight, save_dir):
        image_name = os.path.basename(image_file)
        image_name = image_name.split('.')[0]
        print(np.shape(depth_attention_weight))
        print(np.shape(soft_attention_weight))

        image = plt.imread(image_file)
        for idx, word in enumerate(caption.split()):
            im = plt.imshow(image)
            depth_viz = self._visualize_depth_attention(depth_attention_weight[idx])
            spatial_viz = self._visualize_soft_attention(soft_attention_weight[idx])

            depth_viz = depth_viz[np.newaxis, np.newaxis]
            spatial_viz = np.tile(np.expand_dims(spatial_viz, 2), 3)
            attention_map = spatial_viz * depth_viz
            plt.imshow(attention_map, alpha=0.7, extent=im.get_extent())
            plt.title(word)

            plt.savefig(os.path.join(save_dir,
                                     '{}_{}_result.jpg'.format(image_name, idx)))

    def _format_caption(self, caption):
        words = caption.split()

        if words[-1] == self.end_token:
            words[-1] = '.'
        elif words[-1] != '.':
            words.append('.')
        sentence = "".join(
            [" " + w if not w.startswith("'") and w not in string.punctuation else w for w in words]).strip()
        return sentence

    def _visualize_depth_attention(self, weight):
        idx = np.argmax(weight)
        color = self._pseudocolor(idx, 0, len(weight))

        return np.array(color)

    def _visualize_soft_attention(self, weight):
        size = int(np.sqrt(len(weight)))
        weight = np.reshape(weight, [size, size])
        n_upsample = self.image_shape[0] / size
        upsampled_weight = weight.repeat(n_upsample, axis=0).repeat(n_upsample, axis=1)
        upsampled_weight = gaussian_filter(upsampled_weight, sigma=7)

        return upsampled_weight

    def _pseudocolor(self, val, minval, maxval):
        """ Convert val in range minval..maxval to the range 0..360 degrees which
            correspond to the colors Red and Green in the HSV colorspace.
        """
        h = (float(val - minval) / (maxval - minval)) * 360

        # Convert hsv color (h,1,1) to its rgb equivalent.
        # Note: hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
        r, g, b = hsv_to_rgb(h / 360, 1., 1.)
        return r, g, b


class CaptionData(object):
    def __init__(self, sentence, memory, output, probs, depth_attention_weights=[], soft_attention_weights=[]):
        self.sentence = sentence
        self.memory = memory
        self.output = output
        self.probs = probs
        self.depth_attention_weights = depth_attention_weights
        self.soft_attention_weights = soft_attention_weights

    @property
    def score(self):
        score = np.sum(np.log(self.probs))
        score = score * (1 / (len(self.probs)**0.7))
        return score

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

