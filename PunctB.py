import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


if __name__ == "__main__":
    tips = sns.load_dataset('tips')
    print(tips)
    print(tips.shape)
    print(tips.dtypes)
    print(tips.describe())
    print(tips.groupby('day')['tip'].mean(numeric_only=True))
    print(tips.groupby('sex')['tip'].mean(numeric_only=True))
    tips_cpy = tips.copy()
    tips_cpy['procent_bacsis'] = (tips_cpy['tip']/tips_cpy['total_bill'])*100
    print(tips_cpy)
    print(tips_cpy.sort_values(by='procent_bacsis',ascending=False))
    print(tips.groupby('day')['smoker'].value_counts())
    print(tips.loc[tips['sex'] == 'Male'].groupby('total_bill'))
