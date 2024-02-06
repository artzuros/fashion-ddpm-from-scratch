import matplotlib.pyplot as plt
import torch

def show_images(images, title =""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

        fig = plt.figure(figsize =(10, 10))
        rows = int(len(images)**(1/2))
        cols = round(len(images)/rows)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)
                if idx < len(images):
                    plt.imshow(images[idx][0], cmap ="gray")
                    idx += 1
        fig.suptitle(title, fontsize = 30)
        plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break