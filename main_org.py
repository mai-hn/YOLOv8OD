import correction_camera
import distance
import yolo
import api
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='./videos/test.mp4', help='path to video')
parser.add_argument('--target_path', type=str, default='../vue/src/assets/video', help='path to video')
parser.add_argument('--model', type=str, default='./weight/best.pt', help='yolov8 model')
parser.add_argument('--knownWidth', type=float, default=50, help='known width of the object')
parser.add_argument('--focalLength', type=float, default=50, help='focal length of the camera')
parser.add_argument('--api_ip', type=str, default='192.168.32.14', help='url of the api')
parser.add_argument('--api_port', type=str, default='30002', help='url of the api')

opt = parser.parse_args()

# 修正相机畸变
camera_matrix = np.array([[906.77436812,0,497.52336392],
                          [0,905.88720158,375.05349837],
                          [0,0,1]])
dist_coeffs = np.array([])
correction_camera.undistort_video(opt.video_path, './videos/test_undistort.mp4', camera_matrix, dist_coeffs)
# 检测目标
yolo.yolov8('./videos/test_undistort.mp4', opt.model, opt.target_path)
def convert_avi_to_mp4(input_file, output_file):
    ffmpeg_cmd = ['ffmpeg', '-i', input_file,'-vcodec', 'h264' , output_file]
    subprocess.run(ffmpeg_cmd)
convert_avi_to_mp4('../vue/src/assets/video/test.avi', '../vue/src/assets/video/result.mp4')
# 计算距离
distance.distance_to_camera(opt.knownWidth, opt.focalLength, './runs/detect/predict/labels', './videos/test_undistort.mp4')
# 辅助目标研判
#api.get_api(opt.api_ip, opt.api_port, './runs/detect/predict/crops')
