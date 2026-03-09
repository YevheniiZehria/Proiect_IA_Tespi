import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


if __name__ == "__main__":


    iris = sns.load_dataset('iris')
    sns.pairplot(iris,hue='species',diag_kind='kde')
    plt.suptitle('Pairplot')
  

    fig,axes = plt.subplots(1,4,figsize=(10,8))
    sns.violinplot(data=iris,x='species',y='sepal_length' ,hue='species', split=False,ax=axes[0])
    axes[0].set_title('Violinplot — lungimea sepalei')
    sns.violinplot(data=iris,x='species',y='sepal_width' ,hue='species', split=False,ax=axes[1])
    axes[1].set_title('Violinplot — latimea sepalei')
    sns.violinplot(data=iris,x='species',y='petal_length' ,hue='species', split=False,ax=axes[2])
    axes[2].set_title('Violinplot — lungimea petalei')
    sns.violinplot(data=iris,x='species',y='petal_width' ,hue='species', split=False,ax=axes[3])
    axes[3].set_title('Violinplot — latimea petalei')



    plt.suptitle("Specii", fontsize=14)
    plt.tight_layout()
    plt.show()