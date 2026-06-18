import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from model.wideresnet import WideResNet
from model.simple_cnn import SimpleCNN
from model.loss import FixMatchLoss
from torch.optim import lr_scheduler
from torch.optim import Adam, SGD
from itertools import cycle
import time
import copy
import json
import os
from utils import plot_curves, set_seed

def collate_fn(batch):
    X = torch.stack([
        transforms.ToTensor()(item[0])
        for item in batch
    ])
    y = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    )
    return X, y

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        weak = self.weak_transform(img)
        strong = self.strong_transform(img)
        return weak, strong

def evaluate(model, dataloader, class_names, device, save_dir=None, is_traineval=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    # debug
    # print(torch.bincount(torch.tensor(all_preds),minlength=10))

    acc = accuracy_score(all_labels, all_preds)

    if is_traineval:
        return acc

    assert save_dir is not None, "save_dir must be provided when is_traineval is False"

    print("Overall Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10)) 
    disp.plot(ax=ax, xticks_rotation=45) 
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    print(f"混淆矩阵已保存至: {save_dir}/confusion_matrix.png")

    result_summary = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return result_summary

def main(config: Config):
    set_seed(config.seed)

    class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)

    # 去标签
    labels = np.array(trainset.targets)
    labeled_idx = []
    label_per_class = config.num_labels // config.num_classes
    for c in range(config.num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.setdiff1d(np.arange(len(trainset)), labeled_idx)

    print(f"Total training samples: {len(trainset)}, Labeled samples: {len(labeled_idx)}, Unlabeled samples: {len(unlabeled_idx)}")

    # 弱强增强
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop( # 随机裁剪
            32,
            padding=4,
            padding_mode="reflect"
        ),
        transforms.ToTensor()
    ])
    if config.augment_type == 'randaugment':
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),

            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.2)) # cutout
        ])
    else:
        pass
    # show_data = trainset.data[0]
    # week_data = weak_transform(Image.fromarray(show_data)).permute(1, 2, 0).numpy()
    # strong_data = strong_transform(Image.fromarray(show_data)).permute(1, 2, 0).numpy()
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(show_data)
    # plt.subplot(1, 3, 2)
    # plt.title("Weak Augmentation")
    # plt.imshow(week_data)
    # plt.subplot(1, 3, 3)
    # plt.title("Strong Augmentation")
    # plt.imshow(strong_data)
    # plt.savefig('result/augmentation_example.png')  # 保存图像到文件

    labeled_trainset = torch.utils.data.Subset(trainset, labeled_idx)
    labeled_trainset = LabeledDataset(labeled_trainset, transform=weak_transform)
    unlabeled_trainset = torch.utils.data.Subset(trainset, unlabeled_idx)
    unlabeled_trainset = UnlabeledDataset(unlabeled_trainset, weak_transform=weak_transform, strong_transform=strong_transform)

    labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=config.batch_size, shuffle=True)
    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=config.batch_size*config.mu, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    if config.model_name == 'wideresnet':
        model = WideResNet(depth=config.depth, widen_factor=config.widen_factor, 
                           num_classes=config.num_classes, dropRate=config.dropout_rate).to(config.device)
        ema_model = copy.deepcopy(model)
    elif config.model_name == 'cnn':
        model = SimpleCNN(num_classes=config.num_classes).to(config.device)
        ema_model = copy.deepcopy(model)
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")
    
    criterion = FixMatchLoss(config)

    if config.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    if config.schedule['scheduler'] == 'cosine':
        lr_lambda = lambda k: np.cos(7 * np.pi * k / (16 * config.num_steps))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.schedule['step_size'], gamma=config.schedule['gamma'])

    print(f"device: {config.device}, labeled_batch_nums: {len(labeled_trainloader)}, " 
          f"unlabeled_batch_nums: {len(unlabeled_trainloader)}, test_batch_nums: {len(testloader)}")

    step = 0
    history = {
        "step": [],
        "loss": [],
        "loss_x": [],
        "loss_u": [],
        "test_acc": [],
    }
    best_acc = 0
    best_model_state = None
    pre_loss = 0
    pre_loss_x = 0
    pre_loss_u = 0
    pre_unmask_counts = np.zeros(config.num_classes, dtype=int)

    label_iter = cycle(labeled_trainloader)
    unlabel_iter = cycle(unlabeled_trainloader)

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for step in range(1, config.num_steps + 1):
        start_time = time.time()
        
        X_train, y_train = next(label_iter)
        X_train, y_train = X_train.to(config.device), y_train.to(config.device)
        
        uX_week, uX_strong = next(unlabel_iter)
        uX_week, uX_strong = uX_week.to(config.device), uX_strong.to(config.device)

        model.train()
        # outputs_x = model(X_train)
        # outputs_u_weak = model(uX_week)
        # outputs_u_strong = model(uX_strong)
        input_id = torch.cat([X_train, uX_week, uX_strong], dim=0) # BN 中分别传入可能导致计算混乱
        outputs = model(input_id)
        outputs_x = outputs[:len(X_train)]
        outputs_u_weak, outputs_u_strong = outputs[len(X_train):].chunk(2)
        loss, loss_x, loss_u, counts = criterion(outputs_x, y_train, outputs_u_weak, outputs_u_strong)

        pre_loss += loss.item()
        pre_loss_x += loss_x.item()
        pre_loss_u += loss_u.item()
        pre_unmask_counts += counts.cpu().numpy()
        if step % config.print_step == 0:
            acc = evaluate(model, testloader, class_names, config.device, is_traineval=True) * 100
            if best_acc < acc:
                best_acc = acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, f"{config.checkpoint_dir}/best_model.pth")
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
                with open(f"{config.save_dir}/history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

            avg_loss = pre_loss / config.print_step
            avg_loss_x = pre_loss_x / config.print_step
            avg_loss_u = pre_loss_u / config.print_step

            print(f"Step [{step}/{config.num_steps}], Loss: {avg_loss:.4f}, "
                f"Loss_x: {avg_loss_x:.4f}, Loss_u: {avg_loss_u:.4f}, "
                f"Test Accuracy: {acc:.2f}%, Time: {time.time() - start_time:.2f}s")
            print(f"Unmask counts : {pre_unmask_counts}")

            # 记录历史
            history["step"].append(step)
            history["loss"].append(avg_loss)
            history["loss_x"].append(avg_loss_x)
            history["loss_u"].append(avg_loss_u)
            history["test_acc"].append(acc)

            pre_loss = 0
            pre_loss_x = 0
            pre_loss_u = 0
            pre_unmask_counts = np.zeros(config.num_classes, dtype=int)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 保存训练曲线json
    with open(f"{config.save_dir}/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练曲线数据已保存至: {config.save_dir}/history.json")

    model.load_state_dict(best_model_state)
    result_summary = evaluate(model, testloader, class_names, config.device, save_dir=config.save_dir)

    with open(f"{config.save_dir}/test_result.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    plot_curves(history, save_path=f"{config.save_dir}/training_curves.png")
    print(f"训练曲线已保存至: {config.save_dir}/training_curves.png")

if __name__ == '__main__':
    config = Config()
    main(config)