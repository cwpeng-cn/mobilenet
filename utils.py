import matplotlib.pyplot as plt


class LossWriter:
    def __init__(self, save_path):
        self.save_path = save_path

    def add(self, loss, i):
        with open(self.save_path, mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


def plot_loss(txt_path, x_label="iteration", y_label="loss", title="Loss Visualization ", font_size=15,
              save_name="loss.png", legend=None):
    all_i = []
    all_val = []
    with open(txt_path, "r") as f:
        all_lines = f.readlines()
        for line in all_lines:
            sp = line.split(" ")
            i = int(sp[0])
            val = float(sp[1])
            all_i.append(i)
            all_val.append(val)
    plt.figure(figsize=(6, 4))
    plt.plot(all_i, all_val)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    if legend:
        plt.legend(legend, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.show()
