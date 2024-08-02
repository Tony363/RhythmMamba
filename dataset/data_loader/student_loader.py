"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import gc
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from utils import logger


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
            
    def preprocess_video(
        self, 
        vid_path:str,
        config_preprocess:dict
    )->np.ndarray:
        frames = self.read_video(vid_path)
        bvps = np.ones(frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        return frames_clips,bvps_clips
    
    def _getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        item_path = self.inputs[index]["path"]
        data,_= self.preprocess_video(item_path,self.config_data.PREPROCESS)
        if data.shape[0] == 0:
            logger.info(f"NO VIDEOS PROCESSED - {data.shape[0]}")
            return self.__getitem__((index - 1) % self.__len__())
  
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0,1, 4, 2, 3))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (0,4, 1, 2, 3))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        logger.info(f"{item_path}")
        logger.info(f"DATA SHAPE - {data.shape}")
        self.unprocessed_inputs -= 1            

        filename = item_path.split(os.sep)[-1]
        for vid in range(data.shape[0]):
            save_path = os.path.join(self.config_data.CACHED_PATH,f"{filename.split('.mp4')[0]}_{vid}.npy")
            np.save(save_path, np.float32(data[vid]))
        
        return self.config_data.CACHED_PATH
    
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        # label = np.load(self.labels[index])
        data = np.float32(data)
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        # label = np.float32(label)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # logger.info(item_path)
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        # split_idx = item_path_filename.rindex('_')
        
        # Following the previous comments, the filename for example would be 501
        # filename = item_path_filename[:split_idx]
        filename = item_path_filename
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        # chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        chunk_id = filename.split('_')[-1].split('.npy')[0]
        return data, 1, filename, chunk_id
    
    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        processed = []
        cache_records = 'rppg_mamba' if 'mamba' in self.cached_path else 'rppg_former'
        if os.path.exists(data_path.split('videos')[0] + os.sep + cache_records + os.sep + "processed_list.txt"):
            with open(data_path.split('videos')[0] + os.sep + cache_records + os.sep + "processed_list.txt", "r") as f:
                processed = [path.split(".mp4")[0].replace('\n','') for path in f.readlines()]
        data_dirs = [
            path 
            for path in sorted(glob.glob(data_path + os.sep + "subject*"))
            if path.split('.mp4')[0] not in processed
        ]
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        data_dirs = data_dirs[:1000]
        dirs = [
            {
            "index": idx,
            "path": data_dir
            } 
            for idx,data_dir in enumerate(data_dirs)
        ]
        with open(data_path.split('videos')[0] + os.sep + cache_records + os.sep + "processed_list.txt", "a") as f:
            f.write("\n".join(data_dirs)+'\n')
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



    def save_multi_process(self, frames_clips, bvps_clips, filename,lock):
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
        count = 0
        input_path_name_list = []
        # logger.info(f"FC - {frames_clips.shape} BC - {bvps_clips.shape}")
        for i in range(len(bvps_clips)):
            input_path_name = self.cached_path + os.sep + "{}_{}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            with lock:
                np.save(input_path_name, frames_clips[i])
            count += 1
        return input_path_name_list, None

    def save(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            count: count of preprocessed data
        """

        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{}_{}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            np.save(input_path_name, frames_clips[i])
            count += 1
        return input_path_name_list,None


    def seq_list_dict(
        self,
        data_dirs:list,
        config_preprocess,
    )->list:
        """Preprocesses a list of data directories.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        file_list_dict = {}
        for i in tqdm(range(len(data_dirs))):
            saved_filename = data_dirs[i]['path']
            frames = self.read_video(saved_filename)
            bvps = np.ones(frames.shape[0])
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, _ = self.save(frames_clips, bvps_clips, saved_filename.split(os.sep)[-1].replace('.mp4',''))
            file_list_dict[i] = input_name_list
        return file_list_dict

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict,lock):
        """ 
        invoked by preprocess_dataset for multi_process.
        """
        saved_filename = data_dirs[i]['path'].split(os.sep)[-1].replace('.mp4','')
        # logger.info(f"SAVED FILENAME - {data_dirs[i]}")
        # Read Frames
        frames = self.read_video(data_dirs[i]['path'])
        bvps = np.ones(frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, saved_filename,lock)
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
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = ()
        cap = cv2.VideoCapture(video_file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"FPS of video {video_file} = {fps}")

        if fps > 35:
            logger.info(f"Incompatible frame rate: {video_file}, FPS = {fps}")

            selected_frames = StudentLoader.convert_fps(video_file, fps)
            return selected_frames

        if not cap.isOpened():
            logger.info(f"Error: Unable to open video file {video_file}")
            return np.array([])

        try: 
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames += (frame,)

            cap.release()

        except Exception as e:
            logger.info(f"Error reading video file {video_file}: {e}")   
        finally:
            cap.release()     

        # logger.info(f"in read_video: Finished reading {video_file} - Total frames: {len(frames)}")
        return np.array(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
    
    @staticmethod
    def convert_fps(video_file, fps):
        selected_frames = ()
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            logger.info(f"Error: unable to open {video_file}") 
            return np.ones(300) 

        count_to = fps // 30
        counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if counter == count_to:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected_frames += (frame,)
                counter = 0
            else:
                counter += 1

        cap.release()
        return np.array(selected_frames)