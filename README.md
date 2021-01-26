# Shopping
An AI that predicts whether online shopping customers will complete a purchase.

## Background
When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn’t planning to complete the purchase.

## Approach
The task is achieved by building a **Nearest-Neighbour Classifier**. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — the classifier will predict whether or not the user will make a purchase. The data set used to train the classifier is provided by [Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007%2Fs00521-018-3523-0)

## Measuring the Accuracy
Accuracy is measured based on two metrics: **sensitivity** (also known as the *“true positive rate”*) and **specificity** (also known as the *“true negative rate”*). ```Sensitivity``` refers to the proportion of positive examples that were correctly identified: in other words, the proportion of users who did go through with a purchase who were correctly identified. ```Specificity``` refers to the proportion of negative examples that were correctly identified: in this case, the proportion of users who did not go through with a purchase who were correctly identified. 

## Sample
Here is how our classifier performs, given a CSV file containing the data.
```
$ python shopping.py shopping.csv
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```

## License
This project was made under CS50's Introduction to Artificial Intelligence, a course of study by HarvardX.<br>
The course is licensed under a [Creative Commons License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
