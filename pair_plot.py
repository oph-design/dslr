import seaborn as sns
from libft import load_data
import matplotlib.pyplot as plt
import sys


plt.rcParams.update({"font.size": 6})


def main():
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    pair = sns.pairplot(
        data, hue="Hogwarts House", diag_kind="hist", plot_kws=dict(linewidth=0.1)
    )
    pair.figure.set_size_inches(15, 13)
    plt.show()


if __name__ == "__main__":
    main()
