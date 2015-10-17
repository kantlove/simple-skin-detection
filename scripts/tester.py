import numpy as np
import glob
import cv2
import os
import utility as util


class Tester:
    test_folder = ''
    result_folder = ''

    def __init__(self, test_folder, result_folder):
        self.test_folder = test_folder
        self.result_folder = result_folder

    def begin_test(self, mean_0, mean_1, var_0, var_1, bern_lamda):
        print 'Processing test image'
        for file_name in os.listdir(self.test_folder):
            if file_name.endswith('.png') or file_name.endswith('.jpg'):
                file_name_only = file_name
                file_name = os.path.join(self.test_folder, file_name)
                test_img = cv2.imread(file_name)
                result = np.zeros(test_img.shape, np.uint8)

                rows, cols, _ = test_img.shape
                pr_prior = util.bern_k(bern_lamda, 1)  # bern_x for skin state
                # print 'row:{0} col:{1}'.format(rows, cols)
                for i in xrange(rows):
                    for j in xrange(cols):
                        px = test_img[i, j]
                        # print 'pixel: {0}'.format(px)
                        # Bayes' rule
                        pr_likelihood = util.normal_k(mean_1, var_1, px)
                        pr_evidence = np.multiply(util.normal_k(mean_0, var_0, px), util.bern_k(bern_lamda, 0))
                        pr_evidence = np.add(pr_evidence, np.multiply(util.normal_k(mean_1, var_1, px), pr_prior))

                        pr_skin_px = np.divide(np.multiply(pr_likelihood, pr_prior), pr_evidence)
                        # print 'Bayes : {0}'.format(pr_skin_px)
                        array_of_bool = np.greater(pr_skin_px, 0.496)
                        is_skin_px = True
                        for b in np.nditer(array_of_bool):
                            is_skin_px = is_skin_px and b

                        if is_skin_px:
                            util.set_pixel_white(result, i, j)
                        else:
                            util.set_pixel_black(result, i, j)

                cv2.imwrite(os.path.join(self.result_folder, file_name_only + '-rs.png'), result)
                print 'Image named "{0}" is done'.format(file_name)

        print 'Completed!'
