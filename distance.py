import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

def distance_to_camera(knownWidth, focalLength, labels_dir_path, width, height):
    print('calculating distance...')
    # 读取标签文件
    labels_dir_path = pathlib.Path(labels_dir_path)
    for labels_file in tqdm(labels_dir_path.iterdir(), desc='calculating distance', total=len(list(labels_dir_path.iterdir()))):
        if labels_file.suffix == '.txt':
            result_txt = labels_file
            # 读取txt文件
            result = pd.read_csv(result_txt, sep=' ', header=None)
            result = np.array(result)
            w = result[:, 4]    
            w = w * width

            # 计算距离
            distance = (knownWidth * focalLength) / w


            # 将距离写入txt文件
            result = np.concatenate((result, distance.reshape(-1, 1)), axis=1)
            result = pd.DataFrame(result)
            result.to_csv(result_txt, sep='\t', header=None, index=False)
    print('successfully calculated distance!')

if __name__ == '__main__':
    distance_to_camera(0.6, 0.5, './runs/detect/predict/labels/', './videos/test.mp4')