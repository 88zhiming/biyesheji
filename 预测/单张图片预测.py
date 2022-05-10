import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.My_resnet_model_mix import resnet18
from models.vggnet_model import  vgg
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
img_path ="/home/amax/YSPACK/test_sanfenlei/3/0000010.jpg"
model_weight_path = "/tmp/pycharm_project_773/resnet18_冻结参数_使用自己的数据集/models/训练结果/权重存储2022.05.06-12.19.38/model-40.pth"
use_cbam = True
from models.vggnet_model import vgg
# from models.resnet_model import resnet18
activate_silu = False
num_classes = 3  # 分的类别
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path =  '/tmp/pycharm_project_773/resnet18_冻结参数_使用自己的数据集/models/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model

    model = resnet18(num_classes=num_classes, use_cbam= use_cbam, activate_silu = activate_silu).to(device)

    # load model weights
    weights_path = model_weight_path
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
