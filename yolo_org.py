from ultralytics import YOLO
from pathlib import Path

def yolov8(video_path, model, target_path):
    model = YOLO(model)
    results = model.track(video_path, show=False, save_txt=True, save=True, save_conf=True, save_crop=True, tracker="bytetrack.yaml")

    # 创建源文件路径对象
    source_file_1 = Path(video_path)
    source_file_2 = Path('./runs/detect/predict/test_undistort.avi')

    # 创建目标文件路径对象
    target_file_1 = Path(target_path+'/test.mp4')
    target_file_2 = Path(target_path+'/test.avi')
    # 移动文件
    source_file_1.rename(target_file_1)
    source_file_2.rename(target_file_2)
    

if __name__ == '__main__':
    yolov8('./videos/test.mp4', './weight/best.pt')