import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    tema_c_dashboard()
def tema_c_dashboard():
    tips = sns.load_dataset("tips")

    # creare figura cu 4 subploturi
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Analiza datasetului tips", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    for sex in tips["sex"].unique():
        subset = tips[tips["sex"] == sex]
        ax.scatter(subset["total_bill"], subset["tip"], label=sex)

    ax.set_title("Total Bill vs Tip (colorat dupa sex)")
    ax.set_xlabel("Total Bill")
    ax.set_ylabel("Tip")
    ax.legend()

    ax = axes[0, 1]
    sns.boxplot(data=tips, x="day", y="total_bill",
                order=["Thur", "Fri", "Sat", "Sun"], ax=ax)

    ax.set_title("Distribuția total_bill per zi")
    ax.set_xlabel("Ziua")
    ax.set_ylabel("Total Bill")

    ax = axes[1, 0]
    sns.histplot(data=tips, x="tip", hue="time", kde=True, ax=ax)

    ax.set_title("Distributia bacsisului:")
    ax.set_xlabel("Tip")
    ax.set_ylabel("Frecventa")

    ax = axes[1, 1]
    sns.barplot(data=tips, x="day", y="tip",
                order=["Thur", "Fri", "Sat", "Sun"],
                errorbar="ci", ax=ax)

    ax.set_title("Bacsis mediu per zi")
    ax.set_xlabel("Ziua")
    ax.set_ylabel("Tip mediu")

    plt.tight_layout()

    plt.savefig("analiza_tips.png")

    plt.show()


if __name__=="__main__":
    main()