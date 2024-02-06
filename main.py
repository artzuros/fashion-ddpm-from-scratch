import numpy as np
import torch
import random

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import DataLoader

from displayer import show_first_batch, show_images
from utils_DDPM import DDPM, show_forward, generate_new_images
from utils_UNet import UNet

STORE_PATH_FASHION = f"ddpm_model_fashion.pt"
store_path = "weights/ddpm_fashion.pt"
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

no_train = False
fashion = True
batch_size = 128
n_epochs = 20
lr = 0.001

transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
ds_fn = FashionMNIST
dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
loader = DataLoader(dataset, batch_size, shuffle=True)

show_first_batch(loader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # paper waale parameters
ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
# sum([p.numel() for p in ddpm.parameters()])

#forward
show_forward(ddpm, loader, device)

#backward
generated = generate_new_images(ddpm, gif_name="before_training.gif")
show_images(generated, "Images generated before training")

store_path = "weights/ddpm_fashion.pt"
#if not no_train:
#    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

best_model = DDPM(UNet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name="fashion.gif")
show_images(generated, "Final result")

# from IPython.display import Image

# Image(open('fashion.gif').read())