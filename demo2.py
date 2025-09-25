import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm

from argparse import ArgumentParser
import pydantic

from vae import VAE


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KL, BCE, KL


def train(
    model: VAE,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int = 10,
    beta: float = 1.0,
):
    model.train()
    device = get_device()

    for epoch in range(epochs):
        total_loss = 0
        total_bce = 0
        total_kl = 0

        t = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for data, _ in t:
            data = data.to(device)
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, bce, kl = vae_loss(recon_batch, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kl += kl.item()

            t.set_postfix(
                {
                    "Loss": f"{total_loss / ((t.n + 1) * train_loader.batch_size):.4f}",
                    "BCE": f"{total_bce / ((t.n + 1) * train_loader.batch_size):.4f}",
                    "KL": f"{total_kl / ((t.n + 1) * train_loader.batch_size):.4f}",
                }
            )
        len_data = len(train_loader.dataset)  # type: ignore
        avg_loss = total_loss / len_data
        avg_bce = total_bce / len_data
        avg_kld = total_kl / len_data
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.2f}, BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f}"
        )


def test(model: VAE, test_loader: DataLoader, beta: float = 1.0):
    model.eval()
    device = get_device()
    test_loss = 0
    test_bce = 0
    test_kl = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recon_batch, mu, logvar = model(data)
            loss, bce, kl = vae_loss(recon_batch, data, mu, logvar, beta)
            test_loss += loss.item()
            test_bce += bce.item()
            test_kl += kl.item()

    len_dataset = len(test_loader.dataset)  # type: ignore
    test_loss /= len_dataset
    test_bce /= len_dataset
    test_kl /= len_dataset

    print("Test Results:")
    print(f"Total Loss: {test_loss:.4f}")
    print(f"BCE Loss: {test_bce:.4f}")
    print(f"KL Loss: {test_kl:.4f}")


arg = ArgumentParser()


class Config(pydantic.BaseModel):
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    beta: float = 1.0


if __name__ == "__main__":
    arg.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    arg.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    arg.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    arg.add_argument(
        "--beta", type=float, default=1.0, help="Beta for KL divergence term"
    )
    args = Config(**arg.parse_args().__dict__)

    torch.manual_seed(42)

    train_loader, test_loader = get_mnist_dataloaders(args.batch_size)

    # (1, 28, 28) -> 784
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=20).to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Initial Test:")
    test(model, test_loader, beta=args.beta)

    train(model, train_loader, optimizer, epochs=args.epochs, beta=args.beta)

    print("Final Test:")
    test(model, test_loader, beta=args.beta)
