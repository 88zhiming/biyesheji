from tensorflow.keras.preprocessing import image
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import random
import shutil


ImageFile.LOAD_TRUNCATED_IMAGES = True
"""

"""


# test_data_dir = 'E:/pcb_image_data/data_small/test'
# 载入模型
def read_model():
    model = load_model(r'/tmp/pycharm_project_638/densenet_notmig.h5', compile=False)
    print("模型加载成功")
    return model


# 读取多张图片
def read_img_array(img_dir):
    img = []
    for f in os.listdir(img_dir):
        image_path = os.path.join(img_dir, f)
        if os.path.isfile(image_path):
            images = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(images)
            x = np.expand_dims(x, axis=0)
            img.append(x)
    x = np.concatenate([x for x in img])

    # 读取模型进行预测
    model = load_model()
    y = model.predict(x)
    return y


# 单张图片读取，并预测
def read_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # print(x)

    # 归一化
    amin, amax = x.min(), x.max()  # 求最大最小值
    x = (x - amin) / (amax - amin)

    preds = model.predict(x)
    return preds


# 测试数据集读取
def read_test(test_data_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary'
    )
    model = load_model(r'/tmp/pycharm_project_638/model/densenet/densenet_duochidu.h5', compile=False)
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])
    score = model.evaluate(test_generator, steps=1)
    print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    # y = model.evaluate_generator(test_generator, 20, max_q_size=10,workers=1, use_multiprocessing=False)
    # name_list = model.predict_generator.filenames()
    # print(name_list)
    # return y


"""
    迭代读取文件夹下的所有文件，对每一张图片进行预测，并分类到新文件夹下
"""


def read_file_all(data_dir_path, model, out_path):
    for f in os.listdir(data_dir_path):
        image_path = os.path.join(data_dir_path, f)
        # print(f)
        if os.path.isfile(image_path):
            preds = read_model_predict(image_path, model)
            print(f + " " + str(np.argmax(preds[0]) + 1))
            out_path_tem = out_path + "\\" + str(np.argmax(preds[0]) + 1) + "\\"
            print(out_path_tem)
            if (np.argmax(preds[0]) + 1) == 1:
                print('大雨')
                copyFile(image_path, out_path_tem + f)
            elif (np.argmax(preds[0]) + 1) == 2:
                print('中雨')
                copyFile(image_path, out_path_tem + f)
            elif (np.argmax(preds[0]) + 1) == 3:
                print('小雨')
                copyFile(image_path, out_path_tem + f)
            # elif (np.argmax(preds[0]) + 1) == 4:
            #     print('其他1')
            #     copyFile(image_path, out_path_tem + f)
            # elif (np.argmax(preds[0]) + 1) == 5:
            #     print('其他2')
            #     copyFile(image_path, out_path_tem + f)
            # elif (np.argmax(preds[0]) + 1) == 6:
            #     print('其他3')
            #     copyFile(image_path, out_path_tem + f)
            # elif (np.argmax(preds[0]) + 1) == 7:
            #     print('其他4')
            #     copyFile(image_path, out_path_tem + f)

        else:
            read_file_all(image_path, model, out_path)
    return 0, 0


"""
    预测文件夹下的所有文件
"""


def pre_file_all(data_dir_path, model):
    for f in os.listdir(data_dir_path):
        image_path = os.path.join(data_dir_path, f)
        # print(f)
        if os.path.isfile(image_path):
            preds = read_model_predict(image_path, model)
            # print(f + " " + str(np.argmax(preds[0]) + 1))
            if (np.argmax(preds[0]) + 1) == 1:
                print(image_path + '----大雨----' + str(preds[0][np.argmax(preds[0])]))
            elif (np.argmax(preds[0]) + 1) == 2:
                print(image_path + '----中雨----' + str(preds[0][np.argmax(preds[0])]))
            elif (np.argmax(preds[0]) + 1) == 3:
                print(image_path + '----小雨----' + str(preds[0][np.argmax(preds[0])]))
            # elif (np.argmax(preds[0]) + 1) == 4:
            #     print(image_path + '----其他1----' + str(preds[0][np.argmax(preds[0])]))
            # elif (np.argmax(preds[0]) + 1) == 5:
            #     print(image_path + '----其他2----' + str(preds[0][np.argmax(preds[0])]))
            # elif (np.argmax(preds[0]) + 1) == 6:
            #     print(image_path + '----其他3----' + str(preds[0][np.argmax(preds[0])]))
            # elif (np.argmax(preds[0]) + 1) == 7:
            #     print(image_path + '----其他4----' + str(preds[0][np.argmax(preds[0])]))


def copyFile(file_dir, save_dir):
    shutil.copyfile(file_dir, save_dir)


if __name__ == '__main__':
    # img_file = r'C:\Users\11328\Desktop\枪击\3'
    # out_path = r'C:\Users\11328\Desktop\结果'
    # model = read_model()
    # tc, fc = read_file_all(img_file, model, out_path)
    # print(tc)
    # print(fc)"/home/amax/YSPACK/yonglai_test_data/""/home/amax/YSPACK/yonglai_test_data/"
    read_test(r"/home/amax/YSPACK/test_sanfenlei/")
    # model = read_model()
    # pre_file_all(r"/home/ubuntu/毕业设计/源代码/raininess/数据集/用来测试的数据集/3", model)
    # predict = read_model_predict(r"/home/ubuntu/毕业设计/源代码/raininess/数据集/用来测试的数据集/1/0621.jpg", model)
    # print(predict)
    # if (np.argmax(predict[0]) + 1) == 1:
    #     print('预测结果为：大雨,其概率为：' + str(predict[0][np.argmax(predict[0])]))
    # elif (np.argmax(predict[0]) + 1) == 2:
    #     print('预测结果为：中雨，其概率为：' + str(predict[0][np.argmax(predict[0])]))
    # elif (np.argmax(predict[0]) + 1) == 3:
    #     print('预测结果为：小雨，其概率为：' + str(predict[0][np.argmax(predict[0])]))