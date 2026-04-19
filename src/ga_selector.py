import torch 
import numpy as np
import pygad
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class GeneticFeatureSelector:
    def __init__(self, num_features, num_generations=10, sol_per_pop=10, num_parents_mating=4):
        """
        Initializes the Genetic Algorithm.
        - num_generations: How many cycles of evolution to run.
        - sol_per_pop: How many different feature combinations to test per generation.
        """
        self.num_features = num_features
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        
        self.best_features_mask = None
        
        # We use a very fast, simple model to evaluate 'fitness' during evolution
        self.clf = DecisionTreeClassifier(random_state=42)
        
        # Placeholders for training data
        self.X_train = None
        self.y_train = None

    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        This is the core of the GA. It scores how "fit" a specific combination of features is.
        'solution' is a list of 1s and 0s (e.g., [1, 0, 0, 1...]). 1 means keep the feature, 0 means drop it.
        """
        # If the GA accidentally evolves a solution that drops ALL features, assign a terrible score
        if np.sum(solution) == 0:
            return 0.0

        # Extract only the features where the solution array has a '1'
        selected_features = np.where(solution == 1)[0]
        X_subset = self.X_train[:, selected_features]

        # Evaluate this feature subset using 3-fold cross-validation
        scores = cross_val_score(self.clf, X_subset, self.y_train, cv=3, scoring='accuracy')
        
        # The 'fitness' score is the average accuracy of the model using these features
        return scores.mean()

    def fit(self, X, y):
        """Runs the evolutionary process to find the best feature mask."""
        self.X_train = X
        self.y_train = y
        
        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_features,
            gene_space=[0, 1], # Strict rule: genes can only be 0 (drop) or 1 (keep)
            mutation_percent_genes=10, # 10% chance a 0 flips to 1 or vice versa
            suppress_warnings=True
        )
        
        print(f"Starting Genetic Evolution for {self.num_generations} generations...")
        ga_instance.run()
        
        # Extract the winning chromosome
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        self.best_features_mask = solution
        
        print("\n--- Genetic Algorithm Finished ---")
        print(f"Best Fitness (Accuracy): {solution_fitness:.4f}")
        print(f"Reduced features from {self.num_features} down to {np.sum(solution)} highly optimized features.")
        
        return self.best_features_mask

    def transform(self, X):
        """Applies the winning mask to filter the dataset."""
        if self.best_features_mask is None:
            raise ValueError("You must call fit() before transform().")
            
        selected_features = np.where(self.best_features_mask == 1)[0]
        return X[:, selected_features]

if __name__ == "__main__":
    # --- Local Testing Block ---
    from ingestion import load_dataset
    from features import ECGFeatureExtractor
    
    print("Loading data...")
    X_raw, y_raw, classes = load_dataset()
    
    # To save time during this quick test, we'll only use the first 100 images
    X_test_batch = X_raw[:100]
    y_test_batch = y_raw[:100]
    
    if len(X_test_batch) > 0:
        # Extract features
        extractor = ECGFeatureExtractor()
        X_features = extractor.extract(X_test_batch)
        
        # Run GA
        # We set generations low just for testing. In the final pipeline, we will increase this.
        ga = GeneticFeatureSelector(num_features=X_features.shape[1], num_generations=5)
        best_mask = ga.fit(X_features, y_test_batch)
        
        # Apply the mask
        X_optimized = ga.transform(X_features)
        print(f"Final Optimized Dataset Shape: {X_optimized.shape}")