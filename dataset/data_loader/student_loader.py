"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class StudentLoader(BaseLoader):
    
    def __init__(self, name, data_path, config_data):
        """
        Inferencing your own custom video requires some code tweaking. 
        It's essential to incorporate face detection and cropping during preprocessing,
        with the algorithms located in the 'baseloader', which can be directly invoked.
        Additionally, normalization could also impact the model's performance.
        Therefore, it's preferable to maintain the same preprocessing procedures as during training.

        For your reference:

            Add a dataloader for your video, removing the label processing part
            (you can simplify by setting labels to a fixed value).
            Adjust the output in the trainer's test or metrics.

        """
        super().__init__(name, data_path, config_data)
        
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        # label = np.load(self.labels[index])
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        # label = np.float32(label)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, 1, filename, chunk_id
    
    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = sorted(glob.glob(data_path + os.sep + "subject*"))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [
            {
            "index": idx,#int(''.join(data_dir.split('_')[-2:])),#re.search('subject(\d+)', data_dir).group(0), 
            "path": data_dir
            } 
            for idx,data_dir in enumerate(data_dirs)
        ]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = -1
        input_path_name_list = []
        for i in range(len(bvps_clips)):
            count += 1
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            if os.path.exists(input_path_name):
                continue
            np.save(input_path_name, frames_clips[i])
        return input_path_name_list, None

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ 
        invoked by preprocess_dataset for multi_process.
        """
        saved_filename = data_dirs[i]['index']
        # Read Frames
        frames = self.read_video(data_dirs[i]['path'])
        bvps = np.ones(frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)