import os
import unittest
import numpy as np
from mlbol.vision import plt

figure_dir = f"{os.path.dirname(__file__)}/figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)


def model(x, p):
    return x ** (2 * p + 1) / (1 + x ** (2 * p))


class TestVision(unittest.TestCase):
    def test_science_jcp_cmp1d(self):
        pparam = dict(xlabel="Voltage (mV)", ylabel=r"Current ($\mu$A)")
        x = np.linspace(0.75, 1.25, 201)
        with plt.style.context(["science", "jcp-sc-line", "cmp1d"]):
            fig, ax = plt.subplots()
            for p in [10, 15, 20, 30, 50, 100]:
                ax.plot(x, model(x, p), label=p)
            ax.legend(title="Order")
            ax.autoscale(tight=True)
            ax.set(**pparam)
            fig.savefig(f"{figure_dir}/fig_jcp_cmp1d.pdf")
            plt.close()

    def test_science_jcp_im2d(self):
        with plt.style.context(["science", "jcp-sc-plain"]):
            fig, ax = plt.subplots()
            plt.imshow([np.arange(1000)], aspect="auto", cmap="sunset")
            fig.savefig(f"{figure_dir}/fig_jcp_im2d.pdf")
            plt.close()


if __name__ == "__main__":
    unittest.main()
