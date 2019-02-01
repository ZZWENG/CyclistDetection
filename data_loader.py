import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage.io as io
import pickle


class DataLoader(object):
    def __init__(self, data_path='/data/'):
        # fix SSL certificate issues when loading images via HTTPS
        import ssl; ssl._create_default_https_context = ssl._create_unverified_context

        current_dir = os.getcwd()
        self.data_path = current_dir + data_path

        def load_train_attr(self):
            self.train_object_names = np.load(self.data_path + 'train_object_names.npy')
            self.train_object_x = np.load(self.data_path + 'train_object_x.npy')
            self.train_object_y = np.load(self.data_path + 'train_object_y.npy')
            self.train_object_height = np.load(self.data_path + 'train_object_height.npy')
            self.train_object_width = np.load(self.data_path + 'train_object_width.npy')

        def load_val_attr(self):
            self.val_object_names = np.load(self.data_path + 'val_object_names.npy')
            self.val_object_x = np.load(self.data_path + 'val_object_x.npy')
            self.val_object_y = np.load(self.data_path + 'val_object_y.npy')
            self.val_object_height = np.load(self.data_path + 'val_object_height.npy')
            self.val_object_width = np.load(self.data_path + 'val_object_width.npy')

        load_train_attr(self)
        self.train_num = np.shape(self.train_object_names)[0]
        load_val_attr(self)
        self.val_num = np.shape(self.val_object_names)[0]

        with open(self.data_path + 'vg_idx', 'rb') as f:
            vg_idx = pickle.load(f, encoding='latin1')  # maps idx to the path of the image
        with open(self.data_path + 'ground_truth', 'rb') as f:
            ground_truth = pickle.load(f, encoding='latin1')  # number of cyclists in an image
        self.vg_idx = vg_idx
        label_map = {0: -1, 1: 1}
        self.ground_truth = np.array([label_map[int(ground_truth[i] > 0)] for i in sorted(ground_truth.keys())])
        self.train_ground = self.ground_truth[:self.train_num]
        self.val_ground = self.ground_truth[self.train_num:]

    def get_train_image_path(self, idx):
        return self.vg_idx[idx]

    def get_val_image_path(self, idx):
        return self.vg_idx[idx + self.train_num]
    
    def show_examples(self, annotated=False, label=-1):

        def show_image(idx):
            image_path = self.get_train_image_path(idx)
            I = io.imread(image_path)
            plt.axis('off')
            plt.imshow(I)
    
        def show_image_annotated(idx):
            image_path = self.get_train_image_path(idx)
            I = io.imread(image_path)
            plt.axis('off')
            plt.imshow(I)
            ax = plt.gca()
            
            for i in range(np.shape(self.train_object_y[idx])[0]):
                ax.add_patch(Rectangle((self.train_object_x[idx][i], self.train_object_y[idx][i]),
                                        self.train_object_width[idx][i],
                                        self.train_object_height[idx][i],
                                        fill=False,
                                        edgecolor='cyan',
                                        linewidth=1))
        split_idx = np.where(self.train_ground == label)[0]
        idx_list = np.random.choice(split_idx, 3)

        plt.figure(figsize=(15,3))
        for j,i in enumerate(idx_list):
            plt.subplot(1,3,j+1)
            if annotated:
                show_image_annotated(i)
            else:
                show_image(i)
        plt.suptitle('Query Examples')
        plt.show()
