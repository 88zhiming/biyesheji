import os
import json
import torch
from PIL import Image
from torchvision import transforms
from models.My_resnet_model_mix import resnet18
use_cbam = True

from sklearn import metrics
from models.vggnet_model import vgg
# from models.resnet_model import resnet18
activate_silu = False
num_classes = 3  # 分的类别
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root ="/home/amax/YSPACK/TEST_DATA_BYOESELF/1/"
    # imgs_root ="/home/amax/YSPACK/test_sanfenlei"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = '/tmp/pycharm_project_773/resnet18_冻结参数_使用自己的数据集/models/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    # model = resnet18(num_classes=num_classes, use_cbam= use_cbam, activate_silu = activate_silu)
    model=resnet18(num_classes=num_classes)
    # model =  resnet18(num_classes=num_classes)
    # model_name = "vgg16"
    # model = vgg(model_name=model_name, num_classes=3, init_weights=False)
    model.to(device)
    # load model weights
    weights_path = "/tmp/pycharm_project_773/resnet18_冻结参数_使用自己的数据集/models/训练结果/权重存储2022.05.07-11.08.06/model-74.pth"

    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 5  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            # print(predict)
            probs, classes = torch.max(predict, dim=1)
            # print(probs)
            # print(classes)
            # recall_score = metrics.recall_score(1,classes[i], probs)
            # print('--recall_score:', recall_score)
            #///////////////////////////////
            # recall_score = metrics.recall_score(classes, probs)
            # print('--recall_score:', recall_score)
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                # recall_score = metrics.recall_score(1,cla)
                # print('--recall_score:', recall_score)

if __name__ == '__main__':
    main()
