# coding:utf-8
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = "E:\\desktop(zhuomian)\\yolov10-main\\ultralytics\\cfg\\models\\v10\\yolov10n.yaml"
# 数据集配置文件
data_yaml_path = "E:\\desktop(zhuomian)\\yolov10-main\\data\\data2.yaml"
# 预训练模型
pre_model_name = "E:\\desktop(zhuomian)\\yolov10-main\\weights\\yolov10n.pt"

if __name__ == '__main__':
 # 加载预训练模型45c69
 model = YOLOv10("E:\\desktop(zhuomian)\\yolov10-main\\ultralytics\\cfg\\models\\v10\\yolov10n.yaml").load("E:\\desktop(zhuomian)\\yolov10-main\\weights\\yolov10n.pt")
 # 训练模型
 results = model.train(data=data_yaml_path,epochs=10,batch=4,name='train_v18')