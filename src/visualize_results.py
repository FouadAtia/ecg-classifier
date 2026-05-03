import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def generate_comparison_plots():
    # 1. Setup the Data based on your Tournament Results
    models = ['SVM', 'Hard Voting', 'GB', 'Soft Voting', 'RF', 'Stacking']
    accuracies = [84.95, 94.09, 94.09, 94.62, 95.16, 96.24]
    
    df = pd.DataFrame({'Model': models, 'Accuracy (%)': accuracies})
    df = df.sort_values(by='Accuracy (%)')

    # 2. Create the Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Use a color palette to highlight the winner
    colors = ['#A9A9A9' if x < max(accuracies) else '#2ecc71' for x in df['Accuracy (%)']]
    
    ax = sns.barplot(x='Accuracy (%)', y='Model', data=df, palette=colors)
    
    # Add data labels on the bars
    for i in ax.containers:
        ax.bar_label(i, padding=3, fmt='%.2f%%')

    plt.title('Final Model Performance Comparison', fontsize=15, pad=20)
    plt.xlim(80, 100) # Zoom in to show the nuance between high-performing models
    plt.tight_layout()
    plt.savefig('models/v1.0/accuracy_comparison.png')
    print("✅ Saved accuracy_comparison.png")

    # 3. Create the Heuristic vs GA Comparison Chart
    search_methods = ['Heuristic (SFS)', 'Genetic Algorithm (GA)']
    features_selected = [8, 651]
    base_acc_avg = [81.81, 91.40] # SFS final accuracy vs GA base average

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar for features
    ax1.set_xlabel('Search Method')
    ax1.set_ylabel('Features Selected', color='tab:blue')
    ax1.bar(search_methods, features_selected, color='tab:blue', alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Line for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:red')
    ax2.plot(search_methods, base_acc_avg, color='tab:red', marker='o', linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Search Strategy Impact: Features vs. Performance')
    plt.tight_layout()
    plt.savefig('models/v1.0/feature_search_comparison.png')
    print("✅ Saved feature_search_comparison.png")

if __name__ == "__main__":
    generate_comparison_plots()