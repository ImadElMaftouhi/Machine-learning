
Classical Machine Learning refers to traditional algorithms that predate deep learning and the current wave of AI. Classical ML techniques include various algorithms, such as logistic regression, support vector machines (SVMs), and decision trees. However, these techniques rely heavily on feature engineering to work well. Classical ML often requires human knowledge expertise to generate features. Traditional ML are often easier to understand and interpret than deep learning algorithms or AI models, the simpler algorithms and structures used in traditional models make it easier to understand the relationship between input features and output predictions.

## Main Categories

Classical ML divides into three primary learning paradigms, each suited to different data scenarios.

- **Supervised Learning**: Uses labeled data (input-output pairs) to train models for prediction or classification.
- **Unsupervised Learning**: Works with unlabeled data to find hidden patterns, structures, or groupings.
- **Reinforcement Learning**: Learns through trial-and-error interactions with an environment, maximizing rewards over time.

## Supervised Algorithms

These handle tasks like prediction (regression) or categorization (classification), assuming known outcomes during training.

| Task Type | Algorithms | Use Case Example |
|-----------|------------|------------------|
| Regression | Linear Regression, Logistic Regression, Ridge/Lasso | Predicting house prices or customer spend |
| Classification | Decision Trees, Random Forest, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), Naive Bayes | Spam detection or disease diagnosis |

## Unsupervised Algorithms

Ideal for exploratory analysis on unlabeled data, such as customer segmentation.

- Clustering: K-Means, Hierarchical Clustering, DBSCAN—groups similar data points.
- Dimensionality Reduction: Principal Component Analysis (PCA), t-SNE—simplifies high-dimensional data while preserving structure.
- Association: Apriori—finds rules like "if bread, then butter" in transaction data.

## Reinforcement Algorithms

Less common in basic classical ML but foundational; agents learn policies via rewards.

- Q-Learning, SARSA, Policy Gradient methods—used in robotics or game AI. 
These optimize actions in dynamic environments, like balancing a cart-pole system.

Classical ML excels in interpretability and efficiency on smaller datasets, forming the backbone for many production systems.