import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


if __name__ == "__main__":
    tips = sns.load_dataset('tips')

    fig,axes = plt.subplots(2,2,figsize=(10,8))
    male_data = tips.loc[tips['sex'] =='Male']
    female_data = tips.loc[tips['sex'] =='Female']
    axes[0, 0].scatter(male_data['total_bill'], male_data['tip'], color='blue', label='Male')
    axes[0, 0].scatter(female_data['total_bill'], female_data['tip'], color='red', label='Female')
    axes[0, 0].set_title("Total_bill vs tips per sex")
    axes[0, 0].set_xlabel("Total Bill")
    axes[0, 0].set_ylabel("Tip Amount")
    axes[0, 0].legend()

    sns.boxplot(data=tips,x='day',y='total_bill',ax=axes[0,1])
    axes[0,1].set_title('BoxPlot total_bill per day')
    
    sns.histplot(data=tips,x='tip',hue='time',kde=True,bins=20 , ax=axes[1,0])
    axes[1,0].set_title("HisPlot distributie tip")


    sns.barplot(data=tips,x='day',y='tip',errorbar='ci')
    axes[1,1].set_title("Barplot")
    plt.suptitle("", fontsize=14)
    plt.tight_layout()
    plt.show()
