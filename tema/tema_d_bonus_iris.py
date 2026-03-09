import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
def main():
    cerinta_bonus_iris()
def cerinta_bonus_iris():
    iris = sns.load_dataset("iris")

    pair = sns.pairplot(iris, hue="species", diag_kind="kde")
    pair.fig.suptitle("Pairplot pentru dataset-ul Iris", y=1.02)

    pair.savefig("iris_pairplot.png")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    coloane = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    for i, col in enumerate(coloane):
        sns.violinplot(data=iris,
                       x="species",
                       y=col,
                       hue="species",
                       split=False,
                       ax=axes[i],
                       legend=False)

        axes[i].set_title(col)
        axes[i].set_xlabel("Species")
        axes[i].set_ylabel(col)

    fig.suptitle("Distribuția variabilelor Iris pe specii", fontsize=14)

    plt.tight_layout()

    plt.savefig("iris_violinplots.png")

    plt.show()

if __name__ == "__main__":
    main()