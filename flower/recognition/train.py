import torch
import torch.nn as nn
import torch.optim as optim
from data.data import MyDataset, output_dataset_path_list
from model import *
from torchvision import transforms

def main():
    num_classes = 4
    batch_size = 32
    num_epoch = 20

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    img_path = "data/images"

    tr_list, val_list = output_dataset_path_list(img_path, 4, 0.9)

    print("Setting up data...")
    tr_data = MyDataset(tr_list, transform)
    val_data = MyDataset(val_list, transform)
    tr_dataloader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=len(val_data),
        shuffle=True,
    )

    print("Create model...")
    feature_extract = True
    model_ft = initialize_model(num_classes, feature_extract)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

    print("Start training...")
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_model(
        model_ft,
        tr_dataloader,
        val_dataloader,
        criterion,
        optimizer_ft,
        device,
        num_epochs=num_epoch,
        num_val=len(val_data),
    )

def train_model(
    model,
    tr_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    num_val=0,
):
    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        for phase in ["train", "eval"]:
            if phase == "train":
                model.train()
                dataloaders = tr_dataloader
            else:
                model.eval()
                dataloaders = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for (i, (inputs, labels)) in enumerate(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    print(
                        "({}/{}) Epoch:{}/{} Loss:{:.5f} AveLoss:{:.5f}".format(
                            i,
                            int(len(dataloaders)),
                            epoch,
                            num_epochs,
                            loss.item(),
                            running_loss / (i + 1),
                        )
                    )
                else:
                    print(
                        "phase:eval Loss:{:.5f} Accuracy:{:.5f}%".format(
                            loss.item(), running_corrects.item() / num_val * 100
                        )
                    )
            if (epoch) % 5 == 0:
                print("save model")
                torch.save(model.state_dict(), "save_model/model_{}.pth".format(epoch))
                epoch_loss = running_loss / len(dataloaders.dataset)
                epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print()

if __name__ == "__main__":
    main()
