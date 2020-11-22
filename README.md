# AdaBoost Algorithm

AdaBoost algorithm implementation: apply it to a synthetic dataset as well as a real dataset. The goal is to first implement the algorithm around a very simple decision stub classifier. In a later step this will be replaced by a least squares classifier. To test implementation, use apply script.

# simpleClassifier.py :

Provided training data points are sampled according to their weights(2D synthetic data). It's also posible to generate own synthetic 2D datasets using createDataSet
- Select a simple classifier
- Initialize least error
- Iterate over dimensions
- Calculate thresholds for which we want to check classifier error.
- Candidates for theta are always between two points of different classes.
- Iterate over canidates for theta
- Classify 

# adaboostSimple.py :
Implementation of the AdaBoost algorithm. Evaluate the current boosting classifier in the function eval_adaBoost_simpleClassifier.
- Adaboost with decision stump classifier as weak classifier
- Initialize the classifier models 
- voting weight for each classifier
- Sample data with weights
- Train the weak classifier C_k
- Calculate weighted error for given classifier
- Compute the voting weight for the weak classifier alpha_k
- Update the weights

# eval_adaBoost_simpleClassifier.py :
Evaluate the current boosting classifier in the function eval_adaBoost_simpleClassifier.

# adaboostCross.py :
Next, add a cross-validation step to the training procedure. This means that only a part of the available training data is used for actual training. The remaining part is used to estimate the generalization performance of the learned classifier. To this end, split the training data according to the given percentage (percent = 0.4 means 40% is used for validation). After each iteration k of the algorithm, estimate the classification error of the current boosting classifier (not the base classifier) by cross-validation. Plot the cross validation error estimates vs. the number k of iteration.
- Adaboost with an additional cross validation routine 
- Randomly sample a percentage of the data as test data set
- Initialization
- Initialize the classifier models
- Compute the voting weight for the weak classifier alpha_k
- Update the weights
- Compute error for boosted classifier

# adaboostLSLC.py :

Use a more complex weak classifier in the AdaBoost framework,namely the least squares classifier. Implementation of the AdaBoost algorithm in using least squares classifier. Write the code to evaluate the current boosting classifier in the function eval_adaBoost_leastSquare.
- Adaboost with least squares linear classifier as weak classifier for a D-dim dataset
- Train classifier
- Calculate labeled classification vector
- Compute weighted error of classifier
- Calculate voting weight
- Update weights and normalize

# eval_adaBoost_leastSquare.py :
Compare the classification performance of this classifier learned by AdaBoost with the
one in eval_adaBoost_simpleClassifier

# adaboostUSPS.py :
Used the least-squares based AdaBoost on real data, i.e. the USPS data (provided in usps.mat). The dataset consists of a matrix X and a label vector Y. Each row of the matrix X is an image of size 20 × 14. The first 5000 rows of X contain the images of the digit 2, and the rest contains the images of the digit 9. Perform a random split of the 10000 data points into two equally sized subsets, one for training and one for validation. Run this at least three times and plot the cross validation error estimates vs. the number k of iterations.

- Adaboost with least squares linear classifier as weak classifier on USPS data for a high dimensional dataset
- Sample random a percentage of data as test data set
- Initialization
- initialize loop
- weight sampling of data
- Train the weak classifier Ck
- classify
- calculate error for given classifier
- Compute the voting weight for the weak classifier alphak
- recalculate the weights
- calculate error for boosted classifier

# createDataSet.py :
- Create arbitrary datasets for evaluations
- follow up the code instructions to make a dataset and save it to your local drive



# apply.py :
- Generate or load training data
- Simple weak classifier training (simpleClassifier.py)
- Adaboost using the simple classifiers (adaboostSimple.py, eval_adaBoost_simpleClassifier.py)
  - Compute parameters of K classifiers and the voting weight for each classifier
  - Sample test data from a regular grid to illustrate the decision regions
  - Compute discrete class predictions and continuous class probabilities
  - Show decision surface
  - Visualize logits
- Adaboost with cross-validation (adaboostCross.py)
  - Plot the classification error
  - Sample data from a regular grid to illustrate the decision regions
  - Apply classifier to grid-sampled data
  - Show decision surface
  - Visualize logits
- AdaBoost with least-square classifier compared to simple classifier (adaboostLSLC.py, eval_adaBoost_leastSquare.py)
  - influence of these parameters?
  - Train both classifiers
  - Apply both classifiers on grid data
  - Plot least-square classifier
  - Plot simple classifier
- Least square based AdaBoost with USPS dataset (adaboostUSPS.py)
  - Load USPS dataset
  - Plot error over iterations for multiple runs

