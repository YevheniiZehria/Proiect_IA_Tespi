import seaborn as sns


def main():
    cerinta_3_6()


def cerinta_3_6():
    baxsis = sns.load_dataset("tips")
    print(baxsis)

    print(baxsis.shape)
    print(baxsis.dtypes)
    print(baxsis.describe())
    media_baxis = baxsis.groupby(["day", "sex"])["tip"].mean(numeric_only=True)
    print(media_baxis)

    print("---Cream coloana nou---")
    baxsis_mod = baxsis.copy()
    baxsis_mod["procent_bacsis"] = baxsis_mod["tip"] / baxsis_mod["total_bill"] * 100
    print(baxsis_mod)

    top5 = baxsis_mod.sort_values(by="procent_bacsis", ascending=False).head(5)
    print("Top 5 cele mai generoase mese:")
    print(top5)

    mese_fumatori = baxsis_mod.groupby(["day", "smoker"]).size()
    print("Numărul de mese fumatori pe zi și stare de fumat:")
    print(mese_fumatori)


if __name__ == "__main__":
    main()
