import seaborn as sns
from libft import check_input
import matplotlib.pyplot as plt
import sys


def main() -> None:
    """main function"""
    data = check_input(sys.argv, 0)
    plt.rcParams.update({"font.size": 6})
    pair = sns.pairplot(
        data,
        hue="Hogwarts House",
        # diag_kind="hist",
        plot_kws=dict(linewidth=0.1),
        # diag_kws=dict(bins=10),
        corner=True,
    )
    pair.figure.set_size_inches(15, 13)
    pair.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
