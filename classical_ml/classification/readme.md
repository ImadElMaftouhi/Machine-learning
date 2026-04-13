# Classification

Classification refers to the set of machine learning tasks in which the goal is to predict a discrete label for a given input. Unlike regression, where the output is a continuous value, classification produces a categorical output — for instance, distinguishing whether an email is spam or not spam, or identifying the species of a flower from its measurements.

The classification problem is typically solved by a learning algorithm that takes a collection of labeled examples as input and produces a model. That model can then accept unlabeled examples and return either a predicted label or a probability distribution over the possible labels. The fundamental distinction between classification and regression therefore lies not in the learning procedure itself, but in the nature of the output space: discrete versus continuous.

# Classical Machine Learning Tasks

Classification is one of three primary task families in classical supervised learning, the other two being regression and ranking. Regression predicts a real-valued output, such as the price of a house or the temperature tomorrow. Ranking predicts an ordering over items, as in search engine result relevance. All three share the same supervised structure: labeled training examples, a learned model, and predictions on new data.

Beyond supervised learning, classical machine learning includes two other broad settings. Unsupervised learning operates on unlabeled data, seeking latent structure such as clusters or low-dimensional representations. Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data. An introductory essay on classical methods typically focuses on supervised learning first, because its evaluation is more straightforward and its tasks map directly to many real-world problems.

# Types of Classification

The label assigned to an instance is drawn from a finite set of classes. The structure of that set determines which variant of classification applies. Three common variants are binary classification, multiclass classification, and multilabel classification. They differ in two respects: how many possible classes exist, and whether each instance receives one label or multiple.

## Binary Classification

Binary classification is the simplest case, where the label set contains exactly two classes. A canonical example is spam detection: each email is classified either as "spam" or as "ham" (non-spam). Many real-world decisions reduce to binary outcomes — medical test results (positive or negative), transaction fraud detection (fraudulent or legitimate), or customer churn prediction (will leave or will stay).

## Multiclass Classification

In multiclass classification, the label set contains three or more classes, but each instance is assigned exactly one of them. For example, a handwritten digit recognition system must classify each image as one of ten digits (0 through 9). An image of the digit "7" cannot simultaneously be labeled "3". The defining constraint is mutual exclusivity among the classes.

## Multilabel Classification

Multilabel classification relaxes the exclusivity constraint. Here, each instance may be assigned multiple labels from the set simultaneously. The labels are not mutually exclusive; they represent independent properties or attributes of the instance. For example, an email might be tagged with the departments "Finance", "Marketing", and "Sales" concurrently. Another instance might receive only a subset of these. The output is therefore a set of labels, not a single label.

# What Makes a Method "Classical"

Classical machine learning methods — such as logistic regression, decision trees, support vector machines, and k-nearest neighbors — share several characteristics that distinguish them from modern deep learning. First, they typically require manual feature engineering: the input is not raw data (pixels, audio waveforms, raw text) but a fixed-length vector of hand-constructed features. Second, they have relatively few parameters compared to neural networks, which makes them interpretable and trainable on small datasets. Third, their training algorithms are convex or near-convex in many cases, guaranteeing that the optimization finds a global optimum.

These characteristics are not deficiencies. For tabular data, small sample sizes, or domains where interpretability matters — medical diagnosis, credit scoring, scientific discovery — classical methods often outperform or match deep learning. The choice between classical and deep approaches should be guided by the structure of the data and the requirements of the application, not by fashion.
