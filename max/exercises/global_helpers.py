import matplotlib.pyplot as plt 
import numpy as np 


def image_column(fun, img: np.ndarray, fun_args: list = None, title: str = None, plot_dim: int = 16) -> None:
    fig, ax = plt.subplots(len(fun_args)+1,1,figsize=(len(fun_args)*plot_dim,plot_dim))
    ax[0].imshow(img,cmap="gray")
    ax[0].set_title("original image")
    for i, arg in enumerate(fun_args):
        img_trf = fun(img,arg)
        ax[i+1].imshow(img_trf, cmap="gray")
        ax[i+1].set_title(f"{title} = {arg}")

