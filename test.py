import matplotlib.pyplot as plt
import json
from main import evaluate
from model.simple_cnn import SimpleCNN
import torch
from config import Config
import torchvision
import torchvision.transforms as transforms

def plot_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：loss 曲线
    axes[0].plot(history["step"], history["loss"], label="Total Loss")
    axes[0].plot(history["step"], history["loss_x"], label="Supervised Loss (loss_x)")
    axes[0].plot(history["step"], history["loss_u"], label="Unsupervised Loss (loss_u)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：test accuracy 曲线
    axes[1].plot(history["step"], history["test_acc"], color="tab:green", label="Test Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

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

if __name__ == "__main__":
    config = Config()
    model = SimpleCNN(num_classes=10).to(config.device)
    model_state_path = f"{config.checkpoint_dir}/best_model.pth"
    model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    result_summary = evaluate(model, testloader, class_names, config.device, config.save_dir)

    with open(f"{config.save_dir}/history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    plot_curves(history, save_path=f"{config.save_dir}/training_curves.png")

    with open(f"{config.save_dir}/test_result.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)