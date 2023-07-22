# Computer-efficient Cohen's kappa for all pairwise comparisons

From several hours to several seconds.

[Cohen's kappa coefficient](https://en.wikipedia.org/wiki/Cohen's_kappa) is a measure of inter-rater reliability for categorical items. It is like percent agreement but corrected for random chance. 

Kappa is not implemented in [**scipy**](https://scipy.org/).

Kappa is implemented in [**scikit-learn**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score), but each function call only allows the comparision of 2 'raters'. That makes sense in the context of ML model comparision where you would use kappa to compare ML model output (rater 1) against expected output (rater 2). 

But what if you wanted to measure kappa for each pair of raters among a large set of raters?

In my own research the 'raters' were actually yes/no survey items, and I had 200 of them. Using **scikit-learn**, I would be looping through 39800 (200 x 199) pairs. Each function call took 100 msec on my laptop so all pairwise comparisons would take 1 hour. I had separate data sets to do this analysis on so the whole thing would take several hours.

Several hours can add up if you're exploring your data with different filters.

How can we save some time?

## Matrix operations with simple tricks
Matrix multiplication is one of many linear algebra operations that run super quick in **Numpy** because of the underlying C code. Matrix multiplication is why we can measure the covariance matrix so rapidly among a large set of variables.

Cohen's kappa is not the same as covariance, but we can still use matrix multiplication to quickly obtain a matrix of kappa values. We just have to be crafty.

Kappa requires us to count the number of times the response to 2 items is the same. There are some other things we have to measure but we can use similar tricks there.

Here is a simple case with yes and no responses for 2 items coded as 1s and 0s respectively:

| item 1 | item 2 | agreement | yes-yes | no-no |
|:------:|:------:|:---------:|:-------:|:-----:|
|    0   |    0   |     1     |    0    |   1   |
|    0   |    1   |     0     |    0    |   0   |
|    1   |    0   |     0     |    0    |   0   |
|    1   |    1   |     1     |    1    |   0   |

What we want is a column like *agreement* above whose sum (2) gives us the number of agreements. Note: I am using **Pandas** convention where columns are variables, rather then **Numpy** convention where rows are variables (like with **Numpy.cov()** for covariance). My own code adheres to **Pandas** convention.

Treating *item 1* as a row vector and *item 2*, we can perform matrix multipication to get the sum of yes-yes (1). As an intermediate step in matrix multiplication, our 2 vectors are multiplied value-by-value giving us the yes-yes column above. Summing is the final step of matrix multiplication (1). Agreements can be either yes-yes or no-no so we still need to obtain that sum before we can measure number of agreements. That's easy. We simply flip the values in *item 1* and *item 2* from 0 to 1 and 1 to 0 (new value = absolute value of original value minus 1). Matrix multiplication of these complementary vectors gives us our sum of no-no, whose intermediate step is the no-no column above. Kappa requires some other stuff like number of pairs and number of no-yes and yes-no but similar tricks apply.

The above example is for 2 items but the power of matrix multiplication is that we are multiplying *matrices* -- as many items as we want. And the result is a matrix of all pairwise comparisons. In other words, we'll have a yes-yes matrix for all pairwise comparisions and all the rest. We put all that together and we can calculate all pairwise kappa in seconds.

## The power of Numpy masked arrays
Initially, I thought the above approach was not feasible. Real data has missing values. That is not good for matrix multiplication in **Numpy**. You will end up with a final result of *missing value* for any pair of items where at least one actual missing value occurs!

You could just perform listwise deletion to get rid of data with any missing value. But if different items have missing values for different people then you may end up removing too much data. Some real data is like this:

| item 1 | item 2 | item 3 | item 4 |
|:------:|:------:|:------:|:------:|
|    0   |    0   |   nan  |   nan  |
|    0   |    0   |   nan  |   nan  |
|    1   |    1   |   nan  |   nan  |
|    1   |    1   |   nan  |   nan  |
|    0   |   nan  |    0   |   nan  |
|    0   |   nan  |    1   |   nan  |
|    1   |   nan  |    0   |   nan  |
|    1   |   nan  |    1   |   nan  |
|    0   |   nan  |   nan  |    1   |
|    0   |   nan  |   nan  |    1   |
|    1   |   nan  |   nan  |    0   |
|    1   |   nan  |   nan  |    0   |

List-wise deletion would result in no usable data here!

We could of course loop through each pair and perform list-wise deletion on a pairwise basis. That would make the most possible use of data. But we are back to square one -- several hours of processing instead of several seconds.

Is there still a way we can use matrix multiplication in **Numpy** on messy data like the above and use all of the available data in our calculations?

Yes! We simply have to convert our data to a [Numpy masked array](https://numpy.org/doc/stable/reference/maskedarray.html). Our matrix multiplication approach now works perfectly. It makes the most possible use of the data (missing values are considered on a pairwise basis), and it still runs super fast!

I was actually quite amazed. I had heard of *masked arrays* in passing and knew they existed in **Numpy** but had no real understanding of *why* they existed.

Before working on my current problem, I simply assumed that *masked arrays* existed purely for keeping your data organized. That is a good idea of course. But *masked arrays* are also super useful for efficient computation too! And here's a new example of that.

The full suite of linear algebra operations are available to **Numpy** masked arrays, along with a host of helper functions to assist in organizing/accessing your data. So converting your **Numpy** array to a masked array will be of great value for a wide range of analyses where missing values are involved and you cannot find an existing solution that computes things in 'parallel' (as fast as if it were parallel processing).