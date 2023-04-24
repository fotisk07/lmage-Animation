import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def make_animation(data, show=False):
    if type(data[0, 0]) == 'torch.Tensor':
        data = data.numpy()

    if len(data[0].shape) == 20:
        data = np.transpose(data, (1, 0, 2, 3))

    fig, ax = plt.subplots()
    im = plt.imshow(data[0], cmap='gray', animated=True)

    def init():
        im.set_data(data[0])
        return im,

    def animate(i):
        im.set_data(data[i])
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=20, interval=10, blit=True)
    anim.save("graphs/Animation.gif")

    if show:
        plt.show()


def show_single_frame(data, frame_number):
    if type(data[0, 0]) == 'torch.Tensor':
        data = data.numpy()

    if len(data[0].shape) == 20:
        data = np.transpose(data, (1, 0, 2, 3))

    plt.imshow(data[frame_number], cmap='gray')
