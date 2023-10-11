# -*- coding: utf-8 -*-
import http.client, json
import cv2
import os
import base64
import numpy as np
import pandas
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import io


def init_model(model_path: str):
    model = YOLO(model_path)
    return model


def get_api(url_ip, url_port, dataset_path):
    # 创建结果dataframe
    df = pd.DataFrame(columns=['msg', 'img_id', 'class', 'result', 'process'])
    # 读取数据集
    for dataset_file in os.listdir(dataset_path):
        if dataset_file.endswith('.jpg'):
            # 读取图片
            with open(str(dataset_path + "/" + dataset_file), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # image = base64.b64decode(encoded_string, altchars=None, validate=False)
                # file = open("1.jpg", "wb")
                # file.write(image)
                # file.close()
                conn = http.client.HTTPConnection(url_ip, url_port)  # 接口域名
                # targetType：WZ = 未知，YZ = 已知
                params = json.dumps({'imgBase': encoded_string, 'score': 0.8, 'identifyInfo': 'tank', 'targetType': 'WZ'})
                headers = {'Content-Type': 'application/json;charset=utf-8'}
                # print(params)
                # file = open('example.txt', 'w')
                # file.write(params)
                # file.close()
                # exit(0)
                conn.request('POST', '/ifly/api/target/discern', params, headers)
                tianapi = conn.getresponse()
                result = tianapi.read()
                data = result.decode('utf-8')
                dict_data = json.loads(data)
                print(dict_data)
                # 添加进结果dataframe
                if dict_data['code'] == 200:
                    row = {'msg': dict_data['msg'], 'img_id': dataset_file, 'class': dict_data['data']['class'],
                           'result': dict_data['data']['result'], 'process': dict_data['data']['process']}
                else:
                    row = {'msg': dict_data['msg'], 'img_id': dataset_file, 'class': '', 'result': '', 'process': ''}
                df = df.append(row, ignore_index=True)
                # df = pandas.concat([df, row], ignore_index=True)
                print(df.shape)

    # 保存结果
    df.to_csv('result.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    # yoloV8Model = init_model('./best.pt')
    get_api("192.168.32.14", "30002", './tank_dataset')
