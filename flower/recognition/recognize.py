import cv2
import numpy as np
import torch
from flower.recognition.model import *
from torchvision import transforms
import math

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x) / u

def main(img_path):
    model_path = "flower/recognition/save_model/model_20.pth"
    label = [
        "アメーバ",
        "細菌性　",
        "真菌性　",
        "ウイルス",
    ]

    num_classes = 4
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    model_ft = initialize_model(num_classes, use_pretrained=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    if torch.cuda.is_available():
        model_ft.load_state_dict(torch.load(model_path))
    else:
        model_ft.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    model_ft.eval()

    img_path = "media/" + str(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = transform(img)
    img = img.view(1, 3, 224, 224)

    output = model_ft(img)
    output = output.detach().numpy()
    output = softmax(output)
    indices = np.argsort(-output)
    out_answer = []
    for i in range(4):
        probability = math.floor(100 * output[0, indices[0, i]])  # 確率を四捨五入して整数に変換
        out_answer.append([label[indices[0, i]], probability])
    return out_answer

if __name__ == "__main__":
    result = main()
    for label, probability in result:
        print(f"{label}: {probability}%")
