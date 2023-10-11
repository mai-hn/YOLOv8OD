import cv2
import numpy as np

def undistort_video(input_path, output_path, camera_matrix, dist_coeffs):
    # 读取输入视频
    cap = cv2.VideoCapture(input_path)

    # 获取输入视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建输出视频的写入器
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # 循环读取每一帧并进行畸变矫正
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # 畸变矫正
            #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
            #undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            # 无畸变
            undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)


            # 写入输出视频
            out.write(undistorted_frame)
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_matrix = np.array([[906.77436812,0,497.52336392],
                          [0,905.88720158,375.05349837],
                          [0,0,1]])
    dist_coeffs = np.array([])
    undistort_video('./videos/test.mp4','./videos/test_undistorted.mp4',camera_matrix, dist_coeffs)