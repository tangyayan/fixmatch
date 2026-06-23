import numpy as np
import torch
import matplotlib.pyplot as plt

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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])