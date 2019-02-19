
# Introduction to Decision Trees

## Introduction

In this lesson, we shall look decision tree classifiers. These are rule based classifiers and belong to the first generation of modern AI. Despite the fact that this algorithm has been used in practice for decades, its simplicity and effectiveness for routine classification task is still in par with more sophisticated approaches. In addition, now can combine multiple instances of this algorithm to create much more complex architectures like __random forests__ and other __ensemble__ approcahes. Let's move ahead with this. 

## Objectives

You will be able to:
- Understand and describe a decision tree algorithm in terms of graph architecture
- Describe how decision trees are used to create partitions in s sample space
- Have an overview of the training and prediction stages involved decision tree classification
- Understand the importance of a cost function for decision trees


## Objectives (SG)
YWBAT
* Identify the parts of a decision tree
    * Root node (features)
    * Branches (feature values)
    * Interior nodes (features)
    * Leaf nodes (targets)
* Describe how they are used to create partitions in a sample space ,s.
* Describe the overall process of decision trees.
    * Determine best feature to use as root node using either the Gini Index or Entropy and information gain
    * develops the branches based on its feature value
    * rinse and repeat above to determine the interior nodes depending on our predetermined maximum tree depth
* Describe how the gini coefficient is used with decision trees.
    * the gini coefficient measures how pure of a split a feature creates for our target variable.
    * it helps us determine what feature to start with and order of features to use

## From Graphs to Decision Trees

We have seen basic classification algorithms (a.k.a classifers), including Naive Bayes and signoid based logistic regression earlier. A decision tree is somewhat different type of classifier that performs through a **recursive partition of the sample space**. In this lesson, we shall get a conceptual understanding of how this is achieved. 

A decision tree comprises of decisions that originate from a chosen point in sample space. In terms of a graph theoretic understanding (recall the graph section), it is a **directed acyclic graph with a root called "root node" that has *no incoming edges***. All other nodes have one (and only one) incoming edge. Nodes having outgoing edges are known as **internal**nodes. All other nodes are called **leaves** . Nodes with an incoming edge, but no outgoing edge are called **terminal nodes**. 

>__Directed Acyclic Graphs__

> In computer science and mathematics, a directed acyclic graph (DAG) is a graph that is directed and without cycles connecting the other edges. This means that it is impossible to traverse the entire graph starting at one edge. The graph is a topological sorting, where each node is in a certain order.
![](dtree.png)


## Partitioning the Sample Space

So a decision tree is effectively a DAG as the one seen above where **each internal node partitions the sample space into two (or more) sub-spaces** according to some discrete function of the input attributes values. 

In the simplest and most frequent case, each internal node considers a single attribute so that space is partitioned according to the attribute’s value. In the case of numeric attributes, the condition refers to a range. Let's see a bit more on this with a simple example below.

<img src="dt4.png" width=600>

Above, you can see that root node (testing color) acts as the first decision for feature "Color", creating three new paths. Based on the decision on the color being red green and blue. On the right side, you can see three primary partitions of our sample space. 

If the color is identified as "Red", we don't do any further tests and thus all red objects belong to middle partition without any further sub partition. 

For "Green" color, we do a further test on the attribute "Size". So for green objects, we further classify them into small green and large green objects. On the right we see the green sample space, further divided accordingly 


For "Blue" color, we perform two further tests, if the blue objects are of round shape, we stop there and do not further partition the space. For square blue objects, we perform yet another test and see if they are "small blue square objects" or "large blue square objects". So in the blue partition, we can see that large square and small square are put into their own spaces. Here is another example for a decision tree made for taking decisions on 
bank loan applications.

<img src="dt7.gif" width=600>

So this is the basic idea behind decision trees , every internal node checks for a condition and performs a decision. Every terminal/lead node represents a discrete class. Decision tree induction is closely related to **rule induction**. In essence a decision tree is a just series of IF-ELSE statements (rules). Each path from the root of a decision tree to one of its leaves can be transformed into a rule simply by combining the decisions along the path to form the antecedent part, and taking the leaf’s class prediction as the class value.

## Definition
> A decision tree is a DAG type of classifier where each branch node represents a choice between a number of alternatives and each leaf node represents a classification. An unknown (or test) instance is routed down the tree according to the values of the attributes in the successive nodes. When the instance reaches a leaf, it is classified according to the label assigned to the corresponded leaf.

<img src="dt5.png" width=500>

# What are the parts?
* Root Node (Features)
* Branches (Feature Value Condition)
* Interior Nodes (Features)
* Leaf Nodes (Classes, Labels, Targets, ys)

# What do decision trees do?
* Helps to classify data by ______


# What is a next question:

A real dataset would usually have a lot more features than the example above and will create much bigger trees, but the idea will remain exactly the same. The idea of feature importance is of high importance as selecting the correct feature to make a split that define complexity and effectiveness of the classification process. Regression trees are represented in the same manner, just they predict continuous values like price of a house. 

## Training Process

The process of training a decision tree and predicting the target features of query instances is as follows:

1. Present a dataset of training examples containing features/predictors and a target. (similar to classifiers we have seen earlier)

2. Train the tree model by making splits for the target using the values of predictors. The predictor to use gets selected following the idea of feature selection and uses measures like "__information gain__" and "__gini index__" etc. We shall cover these shortly. 

3. Tree is grown untill some __stopping criteria__ is achieved. This could be a set depth of the tree or any other similar measure. 

4. Show a new set of features to the tree, with an unknown class and let the example propagate through a trained tree. Resulting leaf node represents the class predictions this data. 

<img src="dt6.png" width=900>

## Splitting Criteria

The training process of a decision tree can be generalized as "__Recursive binary Splitting__".  
>In this procedure all the features are considered and different split points are tried and tested using some __Cost Function__. The split with the lowest cost is selected. 

There are couple of algorithms there to build a decision tree:

* __CART (Classification and Regression Trees)__ uses Gini Indexas metric.
    * gi = 1 - (pi/t)^2 - (ni/t)^2
    * gi*(pi+ni)/t
* __ID3 (Iterative Dichotomiser 3)__ uses Entropy function and Information gain as metrics.

Let's quickly see why using these cost criteria is imperative for building a tree. We shall try to develop an intuition using a simple example. Let’s just take a famous dataset in the machine learning world which is weather dataset(playing game Y or N based on weather condition).

<img src="weather.jpeg" width=300>





```python
# Calcualte the gini index for humidity (high normal)
# gini index -> purity of a split (are we dividing play_yes and play_no well?)
# 0 = normal, 1 = high
no_play_humidity = [1, 1, 0, 1, 1] 
yes_play_humidity = [1, 1, 0, 0, 0, 0, 0, 1, 0] 


# P(normal humidity|no play)
no_normal = 0.20 # 0.20^2 = 0.04
# P(high humidity | no play)
no_high = 0.80 # 0.80^2 = 0.64

# P(normal humdity | yes play)
yes_normal = 6.0/9.0
# P(high humidity | yes play)
yes_high = 3.0/9.0


# Weights the larger probabilities
# if g << 1 both numbers must be small (close)
# if g ~ 1 one condition has really high probability
g_nos = no_normal**2 + no_high**2
# g ~ 1 both numbers must be small (close)
# g ~ 0 one condition has really high probability
g_nos = 1 - g_nos

g_yes = yes_normal**2 + yes_high**2
g_yes = 1 - g_yes


# gini ~ 1 this means really really impure split
# gini ~ 0 this means really really pure split
gini = g_nos * (5/14) + g_yes*(9/14)
print("g_humid = {}".format(gini))

# Calculate the gini index for temperature (hot, mild, cool)
# first step split into all categories/values by play
# 1 - hot, 0 - mild, -1 - cold
no_play_temp = [1, 1, -1, 0, 0] # counts
yes_play_temp = [1, 0, -1, -1, -1, 0, 0, 0, 1] # counts

# probability density of each group in the nos
hot_nos = 2/5
mild_nos = 2/5
cold_nos = 1/5

# probability density of each group in the yess
hot_yess = 2/9
mild_yess = 4/9
cold_yess = 3/9

g_nos = hot_nos**2 + mild_nos**2 + cold_nos**2
g_yess = hot_yess**2 + mild_yess**2 + cold_yess**2


g_nos = 1 - g_nos # gini index of each label
g_yess = 1 - g_yess # gini index of each label

g_weighted = g_nos * (5/14) + g_yess*(9/14) # gini index of the feature
print("temp_gini = {}".format(g_weighted)) # higher number indicates more impurity
```

    g_humid = 0.3999999999999999
    temp_gini = 0.6412698412698412


# Gini Index
* G(f) -> We apply the Gini Function (algorithm) to a feature


```python
# Let's calculate it for windy
# Step 1, split your data by labels

no = [0, 1, 1, 0, 1] # these are the windy values for the No's
yes = [0, 0, 0, 1, 0, 0, 1, 1, 0] # these are the windy values for the Yes's

# Step 2, calulate the probability density for each value in each group (no and yes)
p0n = 2/5
p1n = 3/5

p0y = 6/9
p1y = 3/9

# Step 3, calcualte the Gi for each group
gn = 1 - p0n**2 - p1n**2
gy = 1 - p0y**2 - p1y**2

# Step 4, calculate the weighted sum
G = gn*(5/14) + gy*(9/14) 
G
```




    0.4571428571428572




```python
no = [1,1,0,1,1]
yes= [1,1,0,0,0,0,0,1,0]

p0n=1/5
p1n=4/5

p0y=6/9
p1y=3/9

# Step 3, calcualte the Gi for each group
gn = 1 - p0n**2 - p1n**2 # low because p0n was everything it would be 1 - 1 = 0
print(gn)
gy = 1 - p0y**2 - p1y**2
print(gy)

# Step 4, calculate the weighted sum
G = gn*(5/14) + gy*(9/14) 
G
```

    0.31999999999999984
    0.4444444444444445





    0.39999999999999997



So We have four features - X (outlook,temp,humidity and windy) being categorical and one target - y (play Y or N) also categorical, and we need to learn the mapping between X and y. This is a binary classification problem and in order to create a tree, we need to have a root node first and need to decide which feature (outlook,temp,humidity and windy) to use first. Selecting the wrong feature can increase complexity of the tree and it is desired to keep the tree as short as possible. 

## Greedy Search 

We need to determine the attribute that __best__ classifies the training data, we use this attribute at the root of the tree. At each node, we repeat this process creating further splits, until a leaf node is achieved , i.e. all data gets classified.  
> This means we are performing top-down, greedy search through the space of possible decision trees.

In ordert identify the best attribute for ID3 classification trees, we use the "Information Gain" criteria.  Information gain (IG) measures how much “information” a feature gives us about the class. Decision Trees always try to maximize the Information gain. So an attribute with highest Information gain will tested/split first.

Let's move on to the next lesson where we shall look into this criteria with simple examples.

## Additional Resources

* [R2D3:](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). This is highly recommended for getting a visual introduction to decision trees. Excellent animations explaining the training and prediction stages shown above
* [Dataversity: Decision Trees Intro](http://www.dataversity.net/introduction-machine-learning-decision-trees/) A quick and visual introduction to DTs. 

* [Directed Acyclic Graphs](https://cran.r-project.org/web/packages/ggdag/vignettes/intro-to-dags.html). This would help relate early understanding of graph computation to decision tree architectures. 

## Summary 

In this lesson, we saw an introduction to decision trees as simple yet effective classifiers. We looked at how decision trees partition the sample space based by learning rules from a given dataset. We looked at how feature selection for splitting the tree is of such high importance. Next we shall look at Information gain criteria used for feature selection.  
