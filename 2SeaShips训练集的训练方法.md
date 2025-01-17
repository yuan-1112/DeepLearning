## SeaShips7000训练集的训练方法

### 1.环境搭建python3.9，torch2.0.1+cu117，cuda11.7，cudnn8500

### 2.数据集格式转化

> 由于SeaShip数据集的格式不标准，因此需要进行转化

将之前下载的数据集当中的Annotations与JPEGImage移动到新建的data目录下

在同级目录下创建makedata.py，并运行：

```python
import xml.etree.ElementTree as ET
import os
import shutil
import random

xml_file_path = '/data/Annotations\\'  
images_file_path = '/data/JPEGImages\\' 

classes = ["ore carrier", "general cargo ship", "bulk cargo carrier", "container ship", "fishing boat", "passenger ship"]
# 数据集划分比例，训练集75%，验证集15%，测试集15%
train_percent = 0.7
val_percent = 0.15
test_percent = 0.15
# 此处不要改动，只是创一个临时文件夹
if not os.path.exists('data/temp_labels/'):
    os.makedirs('data/temp_labels/')
txt_file_path = 'data/temp_labels/'


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotations(image_name):
    in_file = open(xml_file_path + image_name + '.xml', encoding='UTF-8')
    out_file = open(txt_file_path + image_name + '.txt', 'w', encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        #     continue
        if cls not in classes == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


total_xml = os.listdir(xml_file_path)
num_xml = len(total_xml)  # XML文件总数

for i in range(num_xml):
    name = total_xml[i][:-4]
    convert_annotations(name)


# *********************************************** #
#  parent folder
#  --data
#  ----images
#       ----train
#       ----val
#       ----test
#  ----labels
#       ----train
#       ----val
#       ----test
def create_dir():
    if not os.path.exists('data/images/'):
        os.makedirs('data/images/')
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')

    if not os.path.exists('data/images/train/'):
        os.makedirs('data/images/train')
    if not os.path.exists('data/images/val/'):
        os.makedirs('data/images/val/')
    if not os.path.exists('data/images/test/'):
        os.makedirs('data/images/test/')

    if not os.path.exists('data/labels/train/'):
        os.makedirs('data/labels/train/')
    if not os.path.exists('data/labels/val/'):
        os.makedirs('data/labels/val/')
    if not os.path.exists('data/labels/test/'):
        os.makedirs('data/labels/test/')

    return

# *********************************************** #
# 读取所有的txt文件
create_dir()
total_txt = os.listdir(txt_file_path)
num_txt = len(total_txt)
list_all_txt = range(num_txt)  # 范围 range(0, num)

num_train = int(num_txt * train_percent)
num_val = int(num_txt * val_percent)
num_test = num_txt - num_train - num_val

train = random.sample(list_all_txt, num_train)
# train从list_all_txt取出num_train个元素
# 所以list_all_txt列表只剩下了这些元素：val_test
val_test = [i for i in list_all_txt if not i in train]
# 再从val_test取出num_val个元素，val_test剩下的元素就是test
val = random.sample(val_test, num_val)
# 检查两个列表元素是否有重合的元素
# set_c = set(val_test) & set(val)
# list_c = list(set_c)
# print(list_c)
# print(len(list_c))

print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
for i in list_all_txt:
    name = total_txt[i][:-4]

    srcImage = images_file_path + name + '.jpg'
    srcLabel = txt_file_path + name + '.txt'

    if i in train:
        dst_train_Image = 'data/images/train/' + name + '.jpg'
        dst_train_Label = 'data/labels/train/' + name + '.txt'
        shutil.copyfile(srcImage, dst_train_Image)
        shutil.copyfile(srcLabel, dst_train_Label)
    elif i in val:
        dst_val_Image = 'data/images/val/' + name + '.jpg'
        dst_val_Label = 'data/labels/val/' + name + '.txt'
        shutil.copyfile(srcImage, dst_val_Image)
        shutil.copyfile(srcLabel, dst_val_Label)
    else:
        dst_test_Image = 'data/images/test/' + name + '.jpg'
        dst_test_Label = 'data/labels/test/' + name + '.txt'
        shutil.copyfile(srcImage, dst_test_Image)
        shutil.copyfile(srcLabel, dst_test_Label)
shutil.rmtree(txt_file_path)
```

运行可生成：images和labels两个文件夹，将images和labels放到data文件夹中

### 3. 修改配置文件

在data目录下创建自己的数据集读取配置文件

```python

train: 
  - E:\desktop(zhuomian)\yolov10-main\data\images\train
val: 
  - E:\desktop(zhuomian)\yolov10-main\data\images\test
test: 
  - E:\desktop(zhuomian)\yolov10-main\data\images\val
nc: 6
# Classes
names:
  0: ore carrier
  1: general cargo ship
  2: bulk cargo carrier
  3: container ship
  4: fishing boat
  5: passenger ship
```

使用yolov10m.ymal作为配置文件，使用yolov10m.pt作为权重

**使用trainseaship进行训练**
