import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotting(user_groups):

    total_users = sum(len(users) for users in user_groups.values())

    # Calculate the percentage of each group
    user_percentages = {group: (len(users) / total_users) * 100 for group, users in user_groups.items()}

    # Convert data to DataFrame for Seaborn
    data = pd.DataFrame(list(user_percentages.items()), columns=['', 'Percentage'])

    # Create the plot
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(8, 4))
    bar_plot = sns.barplot(x='', y='Percentage', data=data, palette='viridis')

    # Customize plot
    plt.ylabel("Percentage Of Users", fontsize=14)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=18)  # Increase fontsize of y-ticks
    plt.xticks(fontsize=18)  # Increase fontsize of x-ticks

    # Despine
    sns.despine()

    # Save the plot
    plt.savefig('Yelp_user_type_distribution.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

def plotting_items(item_groups):
    #print(f"item_groups: {item_groups}")
    total_users = sum(len(users) for users in item_groups.values())
    #print(f"total_users: {total_users}")
    # Calculate the percentage of each group
    item_percentages = {group: (len(users) / total_users) * 100 for group, users in item_groups.items()}
    #print(f"item_percentages: {item_percentages}")
    # Convert data to DataFrame for Seaborn
    data = pd.DataFrame(list(item_percentages.items()), columns=['', 'Percentage'])

    # Create the plot
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(8, 4))
    bar_plot = sns.barplot(x='', y='Percentage', data=data, palette='viridis')

    # Customize plot
    plt.ylabel("Percentage Of Each item Group", fontsize=14)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=18)  # Increase fontsize of y-ticks
    plt.xticks(fontsize=18)  # Increase fontsize of x-ticks

    # Despine
    sns.despine()

    # Save the plot
    plt.savefig('Yelp_item_type_distribution.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()
"""import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("DP_Deep_yelp.csv")
ndcg_std_mean =df.groupby('epsilon')['NDCG_test'].agg(['mean','std']).reset_index()
plt.figure(figsize=(10, 6))
plt.errorbar(ndcg_std_mean['epsilon'], ndcg_std_mean['mean'], yerr=ndcg_std_mean['std'], fmt='-o', capsize=5, capthick=1, color='b',ecolor='blue', linestyle='-')

# Labels and title

plt.xlabel('Epsilon')
plt.xticks(ndcg_std_mean['epsilon'])
plt.xscale('log')
plt.ylabel('NDCG')
plt.xticks(ndcg_std_mean['epsilon'])

plt.title('')

plt.grid(True)
plt.savefig('DP_Deep_yelp.pdf', format='pdf', bbox_inches='tight')
plt.show()"""