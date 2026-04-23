

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
    plt.savefig('1M_user_type_distribution.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_result(train_loss, test_loss, save_path=None):

    num_epochs=len(train_loss)
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.plot(range(1, num_epochs + 1), test_loss, label='test', marker='o')
    axs.plot(range(1, num_epochs + 1), train_loss, label='train', marker='x')

    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.set_title('')
    axs.legend()
    axs.grid(True)


    if save_path:
        plt.savefig(save_path)
    else:
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
    plt.ylabel("Percentage Of Each User Group", fontsize=14)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=18)  # Increase fontsize of y-ticks
    plt.xticks(fontsize=18)  # Increase fontsize of x-ticks

    # Despine
    sns.despine()

    # Save the plot
    plt.savefig('1M_item_type_distribution.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    sns.palplot(sns.color_palette())  # This line shows the default palette
    plt.show()



def plot_popularity_distr(popularity):

    frequencies = list(popularity.values())
    frequency_distribution = pd.Series(frequencies).value_counts().sort_index()

    if 0 not in frequency_distribution.index:
        frequency_distribution.loc[0] = 0


    frequency_distribution = frequency_distribution.sort_index()

    x = frequency_distribution.index
    y = frequency_distribution.values


    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='r', label='Popularity Distribution')


    plt.title('Popularity Distribution of Movies', fontsize=16)
    plt.xlabel('Normalized Frequency (Popularity)', fontsize=14)
    plt.ylabel('Number of Movies', fontsize=14)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Display the plot
    plt.show()



