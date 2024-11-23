import pandas as pd
import matplotlib.pyplot as plt

def exercise_0(file):
    return pd.read_csv(file)

def exercise_1(df):
    return list(df.columns)

def exercise_2(df, k):
    return df.head(k)

def exercise_3(df, k):
    return df.sample(k)

def exercise_4(df):
    return df['type'].unique().tolist()

def exercise_5(df):
    return df['nameDest'].value_counts().head(10)

def exercise_6(df):
    return df[df['isFraud'] == 1]

def exercise_7(df):
    pass

def visual_1(df):
    def transaction_counts(df):
        return df['type'].value_counts()

    def transaction_counts_split_by_fraud(df):
        return df.groupby(['type', 'isFraud']).size().unstack()
    fig, axs = plt.subplots(2, figsize=(6, 10))

    transaction_counts(df).plot(ax=axs[0], kind='bar', color='skyblue')
    axs[0].set_title('Transaction Counts by Type')
    axs[0].set_xlabel('Transaction Type')
    axs[0].set_ylabel('Count')

    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar', stacked=True, color=['green', 'red'])
    axs[1].set_title('Transaction Counts by Type, Split by Fraud')
    axs[1].set_xlabel('Transaction Type')
    axs[1].set_ylabel('Count')

    fig.suptitle('Transaction Analysis', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for ax in axs:
        for p in ax.patches:
            ax.annotate(int(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.show()
def visual_2(df):
    def query(df):
        return df[(df['amount'] > 1000) & (df['amount'] < 10000)]
    plot = query(df).plot.scatter(x='amount', y='oldbalanceOrg')
    plot.set_title('Transaction Amount vs Original Balance')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    plt.show()
    return 'Scatter plot created'

def exercise_custom(df):
    pass

def visual_custom(df):
    pass
df = exercise_0('transactions.csv')
df.head()

visual_1(df)
visual_2(df)
