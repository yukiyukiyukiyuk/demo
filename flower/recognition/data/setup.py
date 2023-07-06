import os
import shutil

if __name__ == "__main__":
    label_name = list(open("label.txt"))
    os.mkdir("images")
    for i in range(len(label_name)):
        dir_name = label_name[i].replace("\n", "")
        os.mkdir("images/{}".format(dir_name))
        for k in range(1, 81):
            shutil.move(
                "png/image_{}.png".format(str(k + i * 80).zfill(4)),
                "images/{}/{}.png".format(dir_name, k),
            )
