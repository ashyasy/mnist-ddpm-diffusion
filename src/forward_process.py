import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets as tvdatasets
from torchvision import transforms

from train import DiffusionSampler

import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets as tvdatasets
from torchvision import transforms


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    mnist = tvdatasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # get one MNIST image
    x0, label = mnist[0]          # x0 shape: [1, 32, 32]
    x0 = x0.unsqueeze(0)         # x0 shape: [1, 1, 32, 32]

    timesteps = 500
    cosine_sampler = DiffusionSampler(timesteps, 'cosine')
    linear_sampler = DiffusionSampler(timesteps, 'scaled_linear')

    num_steps = 5
    steps = torch.linspace(0, timesteps - 1, num_steps, dtype=torch.long)

    fig, ax = plt.subplots(2, num_steps, figsize=(num_steps * 2, 4))

    for i in range(num_steps):
        t = torch.tensor([steps[i].item()], dtype=torch.long)

        xt_cosine, _ = cosine_sampler.q_sample(x0, t)
        xt_linear, _ = linear_sampler.q_sample(x0, t)

        ax[0, i].imshow(xt_cosine[0, 0], cmap='gray', vmin=0, vmax=1)
        ax[1, i].imshow(xt_linear[0, 0], cmap='gray', vmin=0, vmax=1)

        ax[0, i].set_title(f'Cosine t={steps[i].item()}')
        ax[1, i].set_title(f'Scaled Linear t={steps[i].item()}')

        ax[0, i].axis('off')
        ax[1, i].axis('off')

    fig.tight_layout()
    fig.savefig('forward_mnist.png', dpi=300)
    plt.show()