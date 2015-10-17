import numpy as np
import glob
import cv2
import os
import utility as util


class Trainer:
    mean_0 = np.zeros(3)
    mean_1 = np.zeros(3)
    var_0 = np.zeros(3)
    var_1 = np.zeros(3)
    bern_lamda = np.zeros(3)
    training_file_ex = ''
    training_folder = ''
    cache_file_path = 'data/cache/data.txt'

    def __init__(self, training_folder, create_new=False):
        self.training_file_ex = os.path.join(training_folder, '*.jpg.png')
        self.training_folder = training_folder

        if create_new or not os.path.exists(self.cache_file_path):
            self.train()
            self.save_cache()
        else:
            self.load_cache()

    def load_cache(self):
        with open(self.cache_file_path, 'r') as cache:
            self.mean_0 = np.fromstring(cache.readline(), dtype=float, sep=' ')
            self.mean_1 = np.fromstring(cache.readline(), dtype=float, sep=' ')
            self.var_0 = np.fromstring(cache.readline(), dtype=float, sep=' ')
            self.var_1 = np.fromstring(cache.readline(), dtype=float, sep=' ')
            line = cache.readline()
            self.bern_lamda = float(line)

    def save_cache(self):
        np.savetxt(self.cache_file_path, (self.mean_0, self.mean_1, self.var_0, self.var_1))
        with open(self.cache_file_path, 'a') as cache:
            cache.write(str(self.bern_lamda))

    def train(self):
        img_count = 0
        valid_files = 0
        for file_name in os.listdir(self.training_folder):
            if file_name.endswith('.png') and not file_name.endswith('-m.png'):
                valid_files += 1

        print 'Found {0} valid images.'.format(valid_files)
        print 'Begin training!'

        for file_name in os.listdir(self.training_folder):
            if not file_name.endswith('.png') or file_name.endswith('-m.png'):
                continue
            file_name = os.path.join(self.training_folder, file_name)
            image = cv2.imread(file_name)
            solution = cv2.imread(file_name.replace('.png', '-m.png'))

            p = image.shape
            rows, cols, _ = image.shape
            n_pixels = rows * cols * 1.0  # avoid integer division
            n_pixels_is_skin = 0  # used to calculate bern_lamda

            # calculate means
            for i in xrange(rows):
                for j in xrange(cols):
                    px = image[i, j]
                    px_sol = solution[i, j]
                    is_skin = util.is_skin(px_sol)
                    if is_skin:
                        self.mean_1 = np.add(self.mean_1, np.divide(px, n_pixels))
                        n_pixels_is_skin += 1
                    else:
                        self.mean_0 = np.add(self.mean_0, np.divide(px, n_pixels))
            # print 'mean(skin):{0} / mean(no skin):{1}'.format(mean_1, mean_0)

            # calculate variance
            for i in xrange(rows):
                for j in xrange(cols):
                    px = image[i, j]
                    px_sol = solution[i, j]
                    is_skin = util.is_skin(px_sol)
                    if is_skin:
                        self.var_1 = np.add(self.var_1, np.divide(np.power(np.subtract(px, self.mean_1), 2), n_pixels))
                    else:
                        self.var_0 = np.add(self.var_0, np.divide(np.power(np.subtract(px, self.mean_0), 2), n_pixels))
            # print 'variance(skin):{0} / variance(no skin):{1}'.format(var_1, var_0)

            self.bern_lamda = n_pixels_is_skin / n_pixels
            # print 'Bernoulli dist lambda: {0}'.format(bern_lamda)
            print 'Image {0} done.'.format(img_count)

            img_count += 1

        print 'Complete! {0} images read\n'.format(img_count)
