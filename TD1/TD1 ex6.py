"""
Exercise 6: Advanced Applications - Hybrid Recommendation Algorithm
Combining multiple similarity measures with optimization techniques
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class HybridRecommendationSystem:
    """
    Advanced recommendation system combining multiple similarity measures
    """
    
    def __init__(self, alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.34):
        """
        Initialize the hybrid recommendation system
        
        Args:
            alpha: Weight for Jaccard similarity
            beta: Weight for cosine similarity  
            gamma: Weight for Pearson correlation
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.user_item_matrix = None
        self.user_similarities = None
        
    def normalize_weights(self):
        """Ensure weights sum to 1"""
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
    
    def jaccard_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate Jaccard similarity between two users
        
        Args:
            u, v: User rating vectors
            
        Returns:
            Jaccard similarity score
        """
        # Convert ratings to binary (rated vs not rated)
        u_binary = (u > 0).astype(int)
        v_binary = (v > 0).astype(int)
        
        intersection = np.sum(u_binary & v_binary)
        union = np.sum(u_binary | v_binary)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate cosine similarity between two users
        
        Args:
            u, v: User rating vectors
            
        Returns:
            Cosine similarity score
        """
        # Handle zero vectors
        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            return 0.0
            
        return 1 - cosine(u, v)
    
    def pearson_correlation(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate Pearson correlation between two users
        
        Args:
            u, v: User rating vectors
            
        Returns:
            Pearson correlation coefficient
        """
        # Only consider items rated by both users
        common_items = (u > 0) & (v > 0)
        
        if np.sum(common_items) < 2:
            return 0.0
            
        u_common = u[common_items]
        v_common = v[common_items]
        
        try:
            correlation, _ = pearsonr(u_common, v_common)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def hybrid_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate hybrid similarity combining all three measures
        
        Args:
            u, v: User rating vectors
            
        Returns:
            Hybrid similarity score
        """
        jaccard = self.jaccard_similarity(u, v)
        cosine_sim = self.cosine_similarity(u, v)
        pearson = self.pearson_correlation(u, v)
        
        # Normalize Pearson correlation to [0, 1] range
        pearson_normalized = (pearson + 1) / 2
        
        hybrid_score = (self.alpha * jaccard + 
                       self.beta * cosine_sim + 
                       self.gamma * pearson_normalized)
        
        return hybrid_score
    
    def fit(self, user_item_matrix: np.ndarray):
        """
        Fit the recommendation system to user-item data
        
        Args:
            user_item_matrix: Matrix where rows are users, columns are items
        """
        self.user_item_matrix = user_item_matrix
        n_users = user_item_matrix.shape[0]
        
        # Calculate pairwise similarities
        self.user_similarities = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity = self.hybrid_similarity(user_item_matrix[i], user_item_matrix[j])
                self.user_similarities[i, j] = similarity
                self.user_similarities[j, i] = similarity
        
        # Set diagonal to 1 (user similarity with themselves)
        np.fill_diagonal(self.user_similarities, 1.0)
    
    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Predict rating for a user-item pair using collaborative filtering
        
        Args:
            user_id: User index
            item_id: Item index
            k: Number of similar users to consider
            
        Returns:
            Predicted rating
        """
        if self.user_similarities is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get similar users
        user_similarities = self.user_similarities[user_id]
        similar_users = np.argsort(user_similarities)[::-1][1:k+1]  # Exclude self
        
        # Calculate weighted average of ratings
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for similar_user in similar_users:
            if self.user_item_matrix[similar_user, item_id] > 0:  # User has rated this item
                similarity = user_similarities[similar_user]
                rating = self.user_item_matrix[similar_user, item_id]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0.0
        
        return weighted_sum / similarity_sum


class OptimizationMethods:
    """
    Various optimization methods for finding optimal weights
    """
    
    @staticmethod
    def greedy_optimization(recommendation_system: HybridRecommendationSystem, 
                           user_item_matrix: np.ndarray, 
                           test_data: np.ndarray,
                           step_size: float = 0.1,
                           max_iterations: int = 100) -> Tuple[float, float, float]:
        """
        Greedy optimization to find optimal weights
        
        Args:
            recommendation_system: The recommendation system to optimize
            user_item_matrix: Training data
            test_data: Test data for evaluation
            step_size: Step size for weight updates
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimal weights (alpha, beta, gamma)
        """
        best_precision = 0.0
        best_weights = (recommendation_system.alpha, 
                       recommendation_system.beta, 
                       recommendation_system.gamma)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try increasing each weight
            for i, weight_name in enumerate(['alpha', 'beta', 'gamma']):
                # Store original weights
                original_weights = (recommendation_system.alpha,
                                  recommendation_system.beta,
                                  recommendation_system.gamma)
                
                # Increase current weight
                if weight_name == 'alpha':
                    recommendation_system.alpha += step_size
                elif weight_name == 'beta':
                    recommendation_system.beta += step_size
                else:
                    recommendation_system.gamma += step_size
                
                # Normalize weights
                recommendation_system.normalize_weights()
                
                # Evaluate
                precision = OptimizationMethods._evaluate_precision(
                    recommendation_system, user_item_matrix, test_data
                )
                
                if precision > best_precision:
                    best_precision = precision
                    best_weights = (recommendation_system.alpha,
                                   recommendation_system.beta,
                                   recommendation_system.gamma)
                    improved = True
                else:
                    # Restore original weights
                    recommendation_system.alpha, recommendation_system.beta, recommendation_system.gamma = original_weights
            
            if not improved:
                break
        
        return best_weights
    
    @staticmethod
    def gradient_descent(recommendation_system: HybridRecommendationSystem,
                        user_item_matrix: np.ndarray,
                        test_data: np.ndarray,
                        learning_rate: float = 0.01,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-6) -> Tuple[float, float, float]:
        """
        Gradient descent optimization for finding optimal weights
        
        Args:
            recommendation_system: The recommendation system to optimize
            user_item_matrix: Training data
            test_data: Test data for evaluation
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimal weights (alpha, beta, gamma)
        """
        weights_history = []
        precision_history = []
        
        for iteration in range(max_iterations):
            # Calculate gradients numerically
            epsilon = 1e-6
            
            # Gradient for alpha
            recommendation_system.alpha += epsilon
            recommendation_system.normalize_weights()
            precision_alpha_plus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.alpha -= 2 * epsilon
            recommendation_system.normalize_weights()
            precision_alpha_minus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.alpha += epsilon  # Restore
            
            grad_alpha = (precision_alpha_plus - precision_alpha_minus) / (2 * epsilon)
            
            # Similar for beta and gamma
            recommendation_system.beta += epsilon
            recommendation_system.normalize_weights()
            precision_beta_plus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.beta -= 2 * epsilon
            recommendation_system.normalize_weights()
            precision_beta_minus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.beta += epsilon  # Restore
            
            grad_beta = (precision_beta_plus - precision_beta_minus) / (2 * epsilon)
            
            recommendation_system.gamma += epsilon
            recommendation_system.normalize_weights()
            precision_gamma_plus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.gamma -= 2 * epsilon
            recommendation_system.normalize_weights()
            precision_gamma_minus = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            recommendation_system.gamma += epsilon  # Restore
            
            grad_gamma = (precision_gamma_plus - precision_gamma_minus) / (2 * epsilon)
            
            # Update weights
            recommendation_system.alpha += learning_rate * grad_alpha
            recommendation_system.beta += learning_rate * grad_beta
            recommendation_system.gamma += learning_rate * grad_gamma
            
            # Normalize weights
            recommendation_system.normalize_weights()
            
            # Track progress
            current_precision = OptimizationMethods._evaluate_precision(
                recommendation_system, user_item_matrix, test_data
            )
            
            weights_history.append((recommendation_system.alpha, 
                                  recommendation_system.beta, 
                                  recommendation_system.gamma))
            precision_history.append(current_precision)
            
            # Check convergence
            if iteration > 0 and abs(precision_history[-1] - precision_history[-2]) < tolerance:
                break
        
        return recommendation_system.alpha, recommendation_system.beta, recommendation_system.gamma
    
    @staticmethod
    def _evaluate_precision(recommendation_system: HybridRecommendationSystem,
                           user_item_matrix: np.ndarray,
                           test_data: np.ndarray) -> float:
        """
        Evaluate precision of the recommendation system
        
        Args:
            recommendation_system: The recommendation system to evaluate
            user_item_matrix: Training data
            test_data: Test data
            
        Returns:
            Precision score
        """
        recommendation_system.fit(user_item_matrix)
        
        predictions = []
        actuals = []
        
        for user_id, item_id, actual_rating in test_data:
            predicted_rating = recommendation_system.predict_rating(user_id, item_id)
            predictions.append(1 if predicted_rating > 3.0 else 0)  # Binary classification
            actuals.append(1 if actual_rating > 3.0 else 0)
        
        if len(set(actuals)) == 1:  # All same class
            return 0.0
            
        return precision_score(actuals, predictions, zero_division=0)


class ModularityOptimization:
    """
    Modularity optimization for community detection
    """
    
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize modularity optimization
        
        Args:
            adjacency_matrix: Graph adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.m = np.sum(adjacency_matrix) / 2  # Total number of edges
        self.degrees = np.sum(adjacency_matrix, axis=1)
        
    def modularity(self, communities: List[List[int]]) -> float:
        """
        Calculate modularity Q for given community structure
        
        Args:
            communities: List of communities, each containing node indices
            
        Returns:
            Modularity value Q
        """
        Q = 0.0
        
        for community in communities:
            for i in community:
                for j in community:
                    if i != j:
                        A_ij = self.adjacency_matrix[i, j]
                        k_i = self.degrees[i]
                        k_j = self.degrees[j]
                        
                        Q += A_ij - (k_i * k_j) / (2 * self.m)
        
        return Q / (2 * self.m)
    
    def greedy_modularity_optimization(self) -> List[List[int]]:
        """
        Greedy modularity optimization algorithm
        
        Returns:
            Optimal community structure
        """
        # Start with each node in its own community
        communities = [[i] for i in range(self.n_nodes)]
        
        improved = True
        while improved:
            improved = False
            best_delta_Q = 0
            best_merge = None
            
            # Try all possible merges
            for i in range(len(communities)):
                for j in range(i + 1, len(communities)):
                    delta_Q = self._calculate_delta_Q(communities[i], communities[j])
                    
                    if delta_Q > best_delta_Q:
                        best_delta_Q = delta_Q
                        best_merge = (i, j)
            
            # Perform best merge if it improves modularity
            if best_delta_Q > 0:
                i, j = best_merge
                communities[i].extend(communities[j])
                communities.pop(j)
                improved = True
        
        return communities
    
    def _calculate_delta_Q(self, community1: List[int], community2: List[int]) -> float:
        """
        Calculate change in modularity when merging two communities
        
        Args:
            community1, community2: Communities to merge
            
        Returns:
            Change in modularity
        """
        delta_Q = 0.0
        
        for i in community1:
            for j in community2:
                A_ij = self.adjacency_matrix[i, j]
                k_i = self.degrees[i]
                k_j = self.degrees[j]
                
                delta_Q += A_ij - (k_i * k_j) / (2 * self.m)
        
        return delta_Q / (2 * self.m)


def generate_synthetic_data(n_users: int = 100, n_items: int = 50, 
                           sparsity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic user-item rating data
    
    Args:
        n_users: Number of users
        n_items: Number of items
        sparsity: Fraction of missing ratings
        
    Returns:
        Tuple of (user_item_matrix, test_data)
    """
    # Generate random ratings
    user_item_matrix = np.random.randint(1, 6, size=(n_users, n_items))
    
    # Introduce sparsity
    mask = np.random.random((n_users, n_items)) < sparsity
    user_item_matrix[mask] = 0
    
    # Create test data (20% of non-zero ratings)
    test_indices = np.where(user_item_matrix > 0)
    test_size = int(0.2 * len(test_indices[0]))
    test_sample = np.random.choice(len(test_indices[0]), test_size, replace=False)
    
    test_data = []
    for idx in test_sample:
        user_id = test_indices[0][idx]
        item_id = test_indices[1][idx]
        rating = user_item_matrix[user_id, item_id]
        
        # Remove from training data
        user_item_matrix[user_id, item_id] = 0
        
        test_data.append((user_id, item_id, rating))
    
    return user_item_matrix, np.array(test_data)


def main():
    """
    Main function demonstrating the hybrid recommendation system
    """
    print("🚀 Exercise 6: Advanced Applications - Hybrid Recommendation System")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    user_item_matrix, test_data = generate_synthetic_data(n_users=50, n_items=30)
    print(f"Training data shape: {user_item_matrix.shape}")
    print(f"Test data size: {len(test_data)}")
    
    # Initialize recommendation system
    print("\n🔧 Initializing hybrid recommendation system...")
    rec_system = HybridRecommendationSystem(alpha=0.4, beta=0.3, gamma=0.3)
    
    # Test individual similarity measures
    print("\n📈 Testing individual similarity measures...")
    user1 = user_item_matrix[0]
    user2 = user_item_matrix[1]
    
    jaccard_sim = rec_system.jaccard_similarity(user1, user2)
    cosine_sim = rec_system.cosine_similarity(user1, user2)
    pearson_sim = rec_system.pearson_correlation(user1, user2)
    hybrid_sim = rec_system.hybrid_similarity(user1, user2)
    
    print(f"Jaccard similarity: {jaccard_sim:.4f}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Pearson correlation: {pearson_sim:.4f}")
    print(f"Hybrid similarity: {hybrid_sim:.4f}")
    
    # Greedy optimization
    print("\n🎯 Running greedy optimization...")
    greedy_weights = OptimizationMethods.greedy_optimization(
        rec_system, user_item_matrix, test_data, step_size=0.05, max_iterations=50
    )
    print(f"Optimal weights (Greedy): α={greedy_weights[0]:.3f}, β={greedy_weights[1]:.3f}, γ={greedy_weights[2]:.3f}")
    
    # Gradient descent optimization
    print("\n📉 Running gradient descent optimization...")
    rec_system_gd = HybridRecommendationSystem(alpha=0.33, beta=0.33, gamma=0.34)
    gd_weights = OptimizationMethods.gradient_descent(
        rec_system_gd, user_item_matrix, test_data, learning_rate=0.01, max_iterations=100
    )
    print(f"Optimal weights (Gradient Descent): α={gd_weights[0]:.3f}, β={gd_weights[1]:.3f}, γ={gd_weights[2]:.3f}")
    
    # Modularity optimization
    print("\n🔗 Testing modularity optimization...")
    # Create a simple graph for demonstration
    n_nodes = 20
    adjacency_matrix = np.random.randint(0, 2, size=(n_nodes, n_nodes))
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) // 2  # Make symmetric
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
    
    modularity_opt = ModularityOptimization(adjacency_matrix)
    communities = modularity_opt.greedy_modularity_optimization()
    modularity_score = modularity_opt.modularity(communities)
    
    print(f"Number of communities found: {len(communities)}")
    print(f"Modularity score: {modularity_score:.4f}")
    print(f"Community sizes: {[len(comm) for comm in communities]}")
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    rec_system_final = HybridRecommendationSystem(*gd_weights)
    rec_system_final.fit(user_item_matrix)
    
    predictions = []
    actuals = []
    for user_id, item_id, actual_rating in test_data[:10]:  # Test on first 10 samples
        predicted_rating = rec_system_final.predict_rating(user_id, item_id)
        predictions.append(predicted_rating)
        actuals.append(actual_rating)
        print(f"User {user_id}, Item {item_id}: Predicted={predicted_rating:.2f}, Actual={actual_rating}")
    
    mse = np.mean([(p - a)**2 for p, a in zip(predictions, actuals)])
    print(f"\nMean Squared Error: {mse:.4f}")
    
    print("\n✅ Exercise 6 completed successfully!")


if __name__ == "__main__":
    main()
