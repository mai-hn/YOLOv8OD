from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
results = model.train(data="./train/tank.yaml", epochs=64, batch=8)  # 训练模型
