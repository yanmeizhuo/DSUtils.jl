# Binary classifier performance evaluation

Whether you are building a binary classifier or need to audit one built
by someone else, there are many things we'd like to know about its performance.
The following sections describe functions that are designed to let you easily get at
commonly used binary classifier performance diagnostic metrics.

The functions are relatively performant and are capable of handling millions of rows
of data.

## kstest

The two sample Kolmogorov-Smirnov test is a statistical test of whether two empirical
distributions are the same.
The test is based on finding the maximum separation between the two cumulative
distribution functions (CDF) and determining the p-value of the test statistic.

For binary classifiers, the predicted probabilities of the two classes should be
different, thus the interest isn't whether the probability distributions
are different, rather, it is how large is the maximal separation and
where does it occur.

Let's generate some data to illustrate the idea.

```@example kstest
using Random, Distributions, Plots

Random.seed!(123)

n100 = rand(Normal(100, 10), 1000)
n100a = rand(Normal(100, 10), 1000)
n120 = rand(Normal(120, 10), 1000)
n140 = rand(Normal(140, 10), 1000)

histogram(n100, nbins = 50, opacity= 0.3)
histogram!(n100a, nbins = 50, opacity= 0.3, legend = nothing)
savefig("kstest-1.svg"); nothing # hide
```

![](kstest-1.svg)

We can use the `kstest` function to find the maximum separation and its location.
The required input is a vector designating the two classes and another vector of
the values, this is the typical data structure of model scoring on
development or validation data.

```@example kstest
using DSUtils

cls = [fill(0, length(n100)); fill(1, length(n100a))]
values = [n100; n100a]
kstest(cls, values)
```

`kstest` returns results in a named tuple:

- `n`, total number of observations
- `n1`, total number of observations in class 1
- `n0`, total number of observations in class 0
- `baserate`, n1 / n, the incidence rate of class 1
- `ks`, the maximum separation between CDF1 and CDF0, a value between [0, 1]
- `ksarg`, argmax, the value where maximum separation is achieved
- `ksdep`, depth of argmax in the sorted values (default sort is from high to low)

ks of 0 means the distributions are indistinguishable,
ks of 1 says the two distributions are complete separable.
These two distributions have negligible separation since they are drawn from the
same distribution.

We now test on moderate separation:

```@example kstest
histogram(n100, nbins = 50, opacity= 0.3)
histogram!(n120, nbins = 50, opacity= 0.3, legend = nothing)
savefig("kstest-2.svg"); nothing    # hide
```

![](kstest-2.svg)

```@example kstest
cls = [fill(0, length(n100)); fill(1, length(n120))]
values = [n100; n120]
kstest(cls, values)
```

There's considerable separation between the classes, and ks is larger than before.

Let's test on widely separately data:

```@example kstest
histogram(n100, nbins = 50, opacity= 0.3)
histogram!(n140, nbins = 50, opacity= 0.3, legend = nothing)
savefig("kstest-3.svg"); nothing    # hide
```

![](kstest-3.svg)

```@example kstest
cls = [fill(0, length(n100)); fill(1, length(n140))]
values = [n100; n140]
kstest(cls, values)
```

We can see that the two classes are nearly separable and
ks is now quite high at 0.949.
These examples illustrate how `ks` can serve as an indicator of the ability to
separate the two classes.

## auroc

A good binary classifier would have high sensitivity
(able to recognize True Positive) and high specificity
(able to recognize True Negatives, hence have low False Positive).
A plot of the trade-off curve of True Positive Rate versus False Positive Rate
at various cutoff probabilities is called the
Receiver Operating Characteristics (ROC) curve.
One way to quantify performance is by the area under the ROC curve,
often abbreviated as AUC or C,
many packages would compute AUC via numeric integration of the ROC curve.
AUC is in the range [0, 1], a perfect model has AUC of 1,
a random model has AUC of 0.5,
and a perfectly backwards model would have AUC of -1.

There is another interpretation of AUC which provides more intuition than
simply as the area under a curve.
If we make all possible pair-wise comparisons between the probabilities of
class 1 with class 0, we can count the incidences of:

- Concordant: class 1 probability > class 0 probability
- Tied: class 1 probability ≈ class 0 probability
- Discordant: class 1 probability < class 0 probability

Then we can compute:

- AUC: (Concordant + 0.5 Tied) / (N1 * N0)
- Gini: 2AUC - 1, or (Concordant - Discordant) / (N1 * N0)
- Goodman-Kruskal Gamma: (Concordant - Discordant) / (Concordant + Discordant),
no penalty for Tied
- Kendall's Tau: (Concordant - Discordant) / (0.5 * (N1+N0) * (N1+N0-1))

We can interpret AUC as the percentage of time class 1 probabilities is larger
than class 0 probabilities (ignoring ties).

The mathematical proof can be found at
[Stack Exchange](https://stats.stackexchange.com/questions/180638/how-to-derive-the-probabilistic-interpretation-of-the-auc)
and
[Professor David J. Hand's article](https://pdfs.semanticscholar.org/1fcb/f15898db36990f651c1e5cdc0b405855de2c.pdf).

```@example kstest
cls = [fill(0, length(n100)); fill(1, length(n140))]
values = [n100; n140]
auroc(cls, values)
```

`auroc` returns results in a named tuple:

- `conc`, number of concordant comparisons
- `tied`, number of tied comparisons
- `disc`, number of discordant comparisons
- `auc`, area under ROC curve, or just area under curve
- `gini`, 2auc - 1

## bcdiag

While `kstest` and `auroc` provide diagnostic measures for comparing
model performance, when there is a model of interest,
it is likely that we need to produce many graphs and table to understand and
document its performance, `bcdiag` allows us to do this easily.

```@example bcd
using Random
using GLM
using DSUtils

function logitgen(intercept::Real, slope::Real, len::Int; seed = 888)
    Random.seed!(seed)
    x = 10 .* rand(len)                     # random uniform [0, 10)
    # sort!(x)                                # x ascending
    logit = @. intercept + slope * x        # logit(prob) = ln(p / (1 + p)) = linear equation
    prob = @. 1. / (1. + exp(-logit))       # probability
    y = rand(len) .<= prob
    y, x
end

m = DataFrame(logitgen(-3, 0.6, 100_000), (:target, :x))
m_logistic = glm(@formula(target ~ x), m, Binomial(), LogitLink())
m.pred = predict(m_logistic)

kstest(m.target, m.pred)
```

```@example bcd
auroc(m.target, m.pred)
```

Running `bcdiag` prints a quick summary:

```@example bcd
mdiag = bcdiag(m.target, m.pred)
```

The output structure allows us to create the following plots and tables to understand:
- the ability of the model to separate the two classes
- the accuracy of the probability point estimates
- how to set cutoff for maximum accuracy
- performance of the model at varying cutoff depth

## ksplot

`ksplot` plots the cumulative distribution of class 1 (true positive rate)
and class 0 (false positive rate) versus depth.

```@example bcd
ksplot(mdiag)
savefig("bcd-ksplot.svg"); nothing # hide
```

It shows where the maximum separation of the two distributions occur.

![](bcd-ksplot.svg)

## rocplot

`rocplot` plots the true positive rate vs. false positive rate (depth is implicit).

```@example bcd
rocplot(mdiag)
savefig("bcd-rocplot.svg"); nothing # hide
```

A perfect model has auc of 1, a random model has auc of 0.5.

![](bcd-rocplot.svg)

## biasplot

Both `ksplot` and `rocplot` rely on the ability of the model to
rank order the observations, the score value itself doesn't matter.
For example, if you took the score and perform any monotonic transform,
`ks` and `auc` wouldn't change.
There are occasions where the score value does matter, where the probabilities
need to be accurate, for example, in expected return calculations.
Thus, we need to understand whether the probabilities are accurate,
`biasplot` does this by plotting the observed response rate versus
predicted response rate to look for systemic bias.
This is also called the *calibration* graph.

```@example bcd
biasplot(mdiag)
savefig("bcd-biasplot.svg"); nothing # hide
```

![](bcd-biasplot.svg)

An unbiased model would lie on the diagnonal, systemic shift off the diagonal
represents over or under estimate of the true probability.

## accuracyplot

People often refer to **(TP + TN) / N** as accuracy of the model,
that is, the ability to correctly identify correct cases.
It is used to compare model performance as well - model with higher accuracy
is a better model.
For a probability based classifier, a cutoff is required to turn probability
to predicted class. So, what is the cutoff value to use to achieve
maximum accuracy?

There are many approaches to setting the best cutoff, one way is to
assign utility values to the four outcomes of [TP, FP, FN, TN] and
maximize the sum across different cutoff's.
Accuracy measure uses the utility values of [1, 0, 0, 1] giving TP + TN.
You can assign negative penalty terms for misclassification as well.

Note that this is different from `kstest` - maximum separation on cumulative
distribution (normalized to 100%) does not account for class size difference,
e.g., class 1 may be only 2% of the cases.

```@example bcd
accuracyplot(mdiag)
savefig("bcd-accuracyplot.svg"); nothing # hide
```

![](bcd-accuracyplot.svg)

## liftcurve

`liftcurve` plots the actual response and predicted response versus depth,
with baserate as 1.

```@example bcd
liftcurve(mdiag)
savefig("bcd-liftcurve.svg"); nothing # hide
```

We can easily see where the model is performing better than average,
approximately the same as average, or below average.

![](bcd-liftcurve.svg)

## cumliftcurve

`cumliftcurve` is similar to `liftcurve`, the difference is it is a plot
of *cumulative* response rate from the top of the model.

```@example bcd
cumliftcurve(mdiag)
savefig("bcd-cumliftcurve.svg"); nothing # hide
```

![](bcd-cumliftcurve.svg)

## liftable

`liftable` is the table from which `liftcurve` is plotted.

```@example bcd
liftable(mdiag)
```

## cumliftable

`cumliftable` is the *cumulative* version of `liftable`.

```@example bcd
cumliftable(mdiag)
```
