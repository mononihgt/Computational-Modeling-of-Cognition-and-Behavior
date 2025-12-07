# **7** Bayesian Parameter Estimation

# Monte Carlo Methods

The goal of this chapter is to introduce the methods that can be applied in cases when the posterior distribution is not analytically tractable. Consider a Beta distribution with parameters *α* and *β*. We know from Equation 6.13 that its mean is *α/(α* + *β)*. But what if we did not know that equation? We could still obtain the mean of a Beta distribution by drawing a huge sample from that distribution in R and computing its mean using the single command mean(rbeta(n,alpha,beta)), where n would be a suitably large number. As long as we can sample from a distribution, given a sufficiently large sample we can begin to understand many attributes of that distribution even if we do not have an explicit formula for it.

This general sampling approach, where an unknown distribution is approximated by a large sample, underlies all Monte Carlo methods of Bayesian inference, although there is considerable variation between methods in assumptions about what we do and do not know about the underlying distributions. Table 7.1 summarizes and compares the main approaches to Bayesian inference and parameter estimation that are presented in this chapter. The figure clarifies that the methods from the previous chapter can only be applied with complete knowledge of all distributions involved. When that knowledge is limited, we can resort to the sampling methods outlined in this chapter.

#### **7.1 Markov Chain Monte Carlo Methods**

The basic idea is simple: we want to replace the unknown posterior distribution with a large sample whose properties mirror those of the posterior, and that we can conveniently interpret to answer questions about the model parameters.

How do we obtain this sample if the posterior is not analytically tractable? The answer is embedded in Bayes' rule provided by Equations 6.9 and 6.20 in the previous chapter. While we may be unable to write down an equation for the posterior and solve it for the parameters, *θ*, we still have access to the right-hand side of the equation. In particular, we necessarily have knowledge of the prior distribution of parameters because we commence our inference by formulating those assumptions. We also typically have knowledge of the likelihood of our observed data given the parameters (though Section 7.3 will relax even that). Those two known terms suffice to compute samples for any particular *θ* in light of the observed data that are at least proportional to the posterior density. The samples are proportional to the posterior because the denominator

| Knowledge required     | Analytic Methods<br>(Chapter 6)                                                 | Monte Carlo Methods<br>(Secon 7.1)                                                      | Approximate Bayesian<br>Computaon (Secon 7.3)                                                                               |
|------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Prior distribuon       | Assumed                                                                         | Assumed                                                                                 | Assumed                                                                                                                     |
| Likelihood             | Computed and known                                                              | Computed and known                                                                      | Cannot be computed but<br>results can be simulated                                                                          |
| Posterior distribution | Derived analytically<br>p(θ y) can be fully<br>•<br>evaluated and<br>integrated | Sampled by MCMC<br>p(θ y) can be<br>•<br>evaluated up to a<br>proporonality<br>constant | Sampled by comparing<br>data to candidate<br>simulaon results<br>neither p(θ y) nor<br>•<br>p(y θ) need to be<br>computable |

**Table 7.1** Summary of all approaches to Bayesian parameter estimation that are discussed in this chapter. The table identifies what it is that must be known or obtainable for each approach.

in Equation 6.9, the evidence or marginal likelihood, is a constant with respect to *θ* (cf. Equation 6.5) and hence can be ignored if we limit our interest to relative comparisons. In recognition of this fact, Bayes' rule is often also written as:

$$P(\boldsymbol{\theta}|y) \propto P(y|\boldsymbol{\theta}) \times P(\boldsymbol{\theta}),$$
 (7.1)

where ∝ stands for "proportional to." Thus, so long as we are able to compute the terms on the right-hand side of Equation 7.1, we can sample from the posterior distribution using a technique known as Markov Chain Monte Carlo, or MCMC.

A Markov Chain is formed by a random process that undergoes transitions between states, where the state at time *t* + 1 depends only on the state at time *t*. Thus, a Markov Chain is ahistorical or "memory-less" because its current state is entirely independent of what happened more than one step in the past. To illustrate, the popular board game *Snakes and Ladders* can be represented by a Markov Chain: At each turn, a piece starts on a given square (i.e., state *t*) and from there has a set of fixed and readily determined probabilities of moving to various other squares (state *t* + 1). However, that set of probabilities for *t* + 1 is entirely independent of how the piece got to the square in state *t* in the first place.

In MCMC, likewise, we probabilistically transition from one state to another, and although successive pairs of states are therefore correlated with each other, states that are more than one step apart are increasingly closer to independence as their separation increases. (In practice, as we shall see in Section 7.2.2, the autocorrelations among successive samples are less of a problem than might appear at first glance.) Several different algorithms have been put forward to instantiate MCMC, and we begin be considering a particularly simple variant.

## 7.1.1 The Metropolis-Hastings Algorithm for MCMC

This algorithm dates back several decades (Hastings, 1970; Metropolis et al., 1953) and is said to be among the top 10 most influential algorithms of the 20th century (Beichl and Sullivan, 2000). The algorithm is readily summarized: we take a guess, add some random noise, and if that improves our guess we accept the answer and move on. If the guess does not get better with noise, we probably reject the answer and proceed with our initial guess. When we have done this many times, our accepted guesses form the desired sample from the posterior.

We formalize this process and provide terminology as follows:

- 1. We create a plausible starting value for the parameter(s) whose posterior distribution – called the *target distribution* for the purposes of MCMC – we wish to estimate. This becomes our current sample.
- 2. We draw random noise from a *proposal distribution* and add it to the current sample to obtain a *proposal*. The proposal distribution must be zero-centered and symmetrical to permit fluctuations in either direction.
- 3. We compare the value of the target distribution that is, a quantity at least proportional to the density or "height" of the posterior – at the proposal to the value of the target distribution at the current sample.
  - a. If the proposal is associated with a greater value than the current sample, accept it.
  - b. If the proposal is associated with a smaller value, accept it with a probability equal to the ratio of the two values, otherwise reject it.
- 4. If the proposal has been accepted, it becomes the next sample in the chain. If the proposal has been rejected, the current sample is reused as the next sample in the chain.
- 5. Return to Step 2 with the next sample as the current sample and continue the process until enough samples have been obtained.

Under some necessary but plausible assumptions, this algorithm will converge on the desired target distribution when sufficient samples are obtained (Andrieu et al., 2003). Note that for the reasons mentioned in connection with Equation 7.1, the values computed in Step 3 only need to be proportional to the posterior. In most instances this means that those values are from an unnormalized distribution (i.e., it need not integrate to 1 as a proper probability density function should; see, e.g., Kruschke 2015, p. 130).

#### A Simple Example in Which the Priors Do Not Matter

We illustrate the Metropolis-Hastings MCMC method with a simplified version that instantiates the original Metropolis version of MCMC. The full Metropolis-Hastings algorithm, which is used in most contemporary applications, includes several extensions; see Chib and Greenberg (1995). We begin with a simple and concrete, if slightly unrealistic, example. The example is similar to one of the cases reported by van Ravenzwaaij et al. (in press) in their brief and highly readable introduction to MCMC. Suppose we administer an intelligence test to a child who is one of a group of children considered to be particularly gifted. This child scores *y* = 144 on a test that is known to be normally distributed with *σ* = 15 for the population at large. What is the likely value of *μ* for the population of gifted children, assuming a uniform prior distribution for *μ*? The reason we assume a uniform prior distribution is because the term *P(θ)* in Equation 7.1 then drops out: A uniform distribution cannot alter the proportionality between the likelihoods and posterior densities.<sup>1</sup>

The solution to this problem based on the Metropolis-Hastings algorithm is shown in Listing 7.1. (Note that this example would be analytically tractable, but there is no harm in using MCMC even when the solution could be computed analytically.)

```
1 chain <− rep (0 ,5000)
2 obs <− 144
3 propsd <− 2 #tuning parameter
4
5 chain [1] <− 150 # starting value
6 for (i in 2 : length (chain) ) {
7 current <− chain[i−1]
8 proposal <− current + rnorm (1 ,0 ,propsd)
9 i f ( dnorm (obs , proposal ,15) > ←	
         dnorm (obs , current ,15) ) {
10 chain[i] <− proposal #accept proposal
11 } else {
12 chain[i] <− ifelse ( runif (1) <
            dnorm (obs , proposal ,15) / dnorm (obs , current ,15) , ←	
13 proposal ,
14 current)
15 }
16 }
```

**Listing 7.1** Metropolis-Hastings estimate of the mean of a normal distribution

Lines 1 through 3 define some necessary variables: We set aside space for the Markov Chain, make a note of the single observed score, and decide on the standard deviation of the proposal distribution. The latter is also known as a *tuning parameter*, and it must not be confused with the standard deviations of the prior or the posterior distributions. We explore the impact of changing the tuning parameter shortly.

We then seed the chain with a plausible value in Line 5. This corresponds to Step 1 in the earlier list of steps involved in the Metropolis-Hastings algorithm.

The loop commencing in Line 6 contains the core of the algorithm. Every iteration involves taking the current sample and turning it into a proposal by adding some random noise with mean zero and standard deviation as defined at the top (variable propsd).

We then compare the values of the target distribution (or heights of the posterior; van Ravenzwaaij et al. in press) for the current sample and the proposal in Line 9. This comparison is using R's built-in function dnorm, which returns normal probability densities. Because in our example the prior plays no role, the desired value of the target distribution is simply the likelihood of the data given the parameters, where the parameters here are the fixed standard deviation (*σ* = 15) defined at the outset, and the current sample or the proposal.

<sup>1</sup> Note that the uniform prior is psychologically unrealistic because it would allow for negative IQs or IQs in the millions. However, for this introductory example we accept this implausibility to keep the computation simple. In reality we would commonly use a Gaussian prior on *μ*.

![](img/_page_172_Figure_2.jpeg)

**Figure 7.1** MCMC output obtained by Listing 7.1 for different parameter values. **a.** Exactly as shown in Listing 7.1. **b.** Starting value set to 10 and the chain limited to 500 iterations. **c.** Starting value is 500, but the standard deviation of the proposal distribution is 20 and the chain runs to 5,000 iterations. In each panel, the top graphs shows the normal probability density function for *μ* = 144, *σ* = 15 (thick gray line), and the posterior density obtained with all MCMC samples included (dashed black line), or the samples from the burn-in period excluded (solid black line). The bottom graph shows the values of the accepted samples across iterations of the chain. The burn-in period is identified in gray.

If the proposal has the higher density (i.e., is closer to the mean of the target distribution), then it is accepted and becomes the next sample in the chain (Line 10). If the proposal has the lower density, it may still be accepted, but only in proportionality to its relative merit: the probability of acceptance is equal to the ratio between the densities of the proposal and the current sample. This comparison and decision is achieved by the ifelse statement commencing in Line 12.

Figure 7.1 contains three panels that summarize posterior estimates for the intelligence test scores of the gifted group of children, under different situations. Panel **a** shows the results from a run of the program exactly as shown in Listing 7.1. The top graph in each panel shows the normal probability density function for *μ* = 144, *σ* = 15 (thick gray line), together with equivalent density estimates based on the samples from the posterior (black dashed line; ignore the solid black line for now).

It can be seen that the MCMC method recovers the expected probability density with reasonable accuracy. The mean of the sampled posterior in Panel **a** is 143.58 and the standard deviation is 16.31. Both values differ little from the known true values of 144 and 15, respectively.

The bottom graph in each panel shows the accepted samples from each iteration across the entire chain. In Panel **a**, the starting value (150; see Line 5 in Listing 7.1) was close to the actual value, and the chain therefore remained reasonably stationary across iterations with most sampled values in the range of 120–160.

The situation looks considerably different in Panel **b**, which shows the results for an MCMC solution with only 500 (as opposed to 5,000) iterations and a starting value of 10 (as opposed to 150). The top of Panel **b** shows that the overall MCMC distribution (dashed lines) is shifted toward the left, with a relatively thick bottom tail. The bottom of the panel shows why: The implausible starting value means that the sampler takes considerable time to converge on the center of the actual distribution. At each iteration, the new proposal will be in relatively close proximity to the current value of the chain because the proposal distribution has a limited standard deviation (in this case, 2). Although proposals that point even further away from the probability mass of the actual posterior are likely to be rejected, it takes a number of iterations before the samples take on reasonable values and the sequence of further samples becomes reasonably stationary. This is a known attribute of all MCMC methods (not just the Metropolis method), and in order to guard against a choice of implausible starting values, the early samples of a chain are usually dismissed. This early period is referred to as "burnin," and the bottom graphs in each panel identify that period in gray. Conventionally, burnin samples are discarded and the MCMC solution consists only of samples obtained after the burnin period. We will take up the issue of burnin again in the next chapter in the context of real-world applications of MCMC. In Panels **a** and **b**, the burnin period consists of 200 iterations. The top of each panel shows the sampled posterior distribution when the burnin is excluded using a solid black line. It can be seen that exclusion of the burnin samples improves the estimate when the starting value is implausible: the solid line in the top of Panel **b** no longer has the thick bottom tail, although its probability mass is still shifted downward compared to the true density. With additional samples this problem would likely disappear; you can verify this yourself by playing with the code in Listing 7.1.

Those problems are resolved in Panel **c**, which shows the results for an even more absurd starting value, namely 500. The reason this extreme starting value nonetheless does not affect the outcome – note how the sampled posterior densities are nearly indistinguishable from the actual density in the top graph – is threefold: First, the number of iterations has been reset to 5,000, which ensures greater stability in the estimate. Second, the burnin period has been extended to include the first 1,000 iterations, thereby giving the chain ample time to reach the central probability mass even if the starting value was an outlier. Third, the standard deviation of the proposal distribution has been increased to 20 (from 2). The bottom graph in the panel shows the effects of that increase in standard deviation: The chain literally jumps from the extreme outlying starting value to the central part of the distribution in a few iterations. Chib and Greenberg (1995), Gelman et al. (1996), and Roberts et al. (1997) discuss suitable values for the standard deviation of the proposal distribution as a function of the nature of the target distribution.

#### Introducing Nonuniform Priors

The preceding example assumed a uniform prior for simplification. We now consider another case involving a normal distribution, but one in which we reintroduce the role of the prior in Equation 7.1. For this example, we are concerned with estimating the average net worth of families (i.e., assets minus liabilities, call that *μ*) in a single county in the United Kingdom. We use statistical information to derive prior expectations, and we assume that the distribution of net worth across counties in the U.K. is normal with a mean of *μ*<sup>0</sup> = *£*326 (in thousands of pounds) and standard deviation *σ*<sup>0</sup> = *£*88 (again in thousands).

Let us now suppose we sample a single household in Somerset at random and ask an accountant to assess the family's net worth. She returns an estimate of *x* = *£*415 and points out that based on her past performance record, her errors are known to be distributed normally with mean zero and standard deviation *sx* = *£*20. What is our best guess about the average net worth of households in Somerset?

Listing 7.2 shows the R code that provides the solution. Much of this is unchanged from Listing 7.1 but there are some crucial differences that merit discussion.

```
1 obs <− 415
2 obssd <− 20
3 priormu <− 326
4 priorsd <− 88
5
6 chain [1] <− 500 # starting value
7 for (i in c (2: length (chain))) {
8 current <− chain[i−1]
9 proposal <− current + rnorm (1 ,0 ,propsd)
10 i f ( ( dnorm (obs , proposal , obssd) ∗
11 dnorm (proposal , priormu , priorsd) ) >
12 ( dnorm (obs , current , obssd) ∗
13 dnorm (current , priormu , priorsd)) ) {
14 chain[i] <− proposal #accept proposal
15 } else {
16 llratio <− ( dnorm (obs , proposal , obssd) ∗
17 dnorm (proposal , priormu , priorsd) ) /
18 ( dnorm (obs , current , obssd) ∗
19 dnorm (current , priormu , priorsd) )
20 chain[i] <− ifelse ( runif (1) < ←	
            llratio , proposal , current)
21 }
22 }
```

**Listing 7.2** Computing posterior normal densities when an informative prior is provided

One important change occurs in Lines 1 through 4, which now also define the parameter values for the prior distribution. Those parameters are called *hyperparameters* because they define the properties of a distribution of parameters. Those hyperparameters are used in the crucial comparisons between the proposal and the current sample in Lines 10 and 16 (each comparison spans multiple lines). Unlike in the earlier program (Listing 7.1), here the comparison of the values of the target distribution for the proposal and current sample is not just based on the likelihood, but is also informed by the prior distribution of parameter values. Those two sources of information are multiplied as per Equation 7.1.

![](img/_page_175_Figure_2.jpeg)

**Figure 7.2** MCMC output obtained by Listing 7.2 for different parameter values. **a.** Exactly as shown in Listing 7.2 with  $\sigma_0=88$ . **b.** Same as **a**, but the standard deviation for the prior distribution is  $\sigma_0=20$ . In each panel, the actual normal probability density function for  $\mu=415, \sigma=20$  is shown in solid gray. The posterior density obtained with all MCMC samples included is shown in dashed black, and the posterior when samples from the burn-in period are excluded is shown in solid black. The dot-dashed gray line shows the normal probability density function for the prior distribution with  $\mu_0=326$  and  $\sigma_0$  as appropriate for each panel.

The results are shown in Figure 7.2. In each panel, the dot-dashed gray line indicates the prior distribution of  $\mu$  and the solid gray line shows the normal probability density function for  $\mu=415$  and  $\sigma=20$ ; that is, for the estimated net worth of our single household and the associated error. The black line indicates the posterior density obtained via MCMC.

Panel **a** displays the results with a very diffuse prior ( $\sigma_0 = 88$ ). The single observation with a smaller error is sufficient to overwhelm the prior, and the posterior density is nearly identical to the gray line – that is a situation in which the prior is uninformative. In Panel **b**, the prior is far less diffuse ( $\sigma_0 = 20$ ) and its standard deviation is identical to that of the single data point. Thus, the gray lines (solid and dot-dashed) outline a distribution with identical spread but in different locations. In consequence, the single observation yields a posterior distribution that sits in between those two extremes and that balances out our prior knowledge and the modest amount of evidence that is provided by our single observation. This result reinforces and amplifies the message of the earlier Figure 6.2, which first introduced the notion that a posterior distribution captures both prior knowledge and the evidence contained in the data (i.e., the likelihood), appropriately weighted by the strength of the evidence.

#### 7.1.2 Estimating Multiple Parameters

Thus far, we have only considered situations involving a single parameter. In reality, of course, most models involve more than a single parameter. Fortunately, the techniques introduced thus far generalize readily to multi-parameter models.

![](img/_page_176_Picture_2.jpeg)

**Figure 7.3** Experimental procedure for a visual working memory task in which participants have to remember the color of a varying number of squares. (Color is here represented in grayscale.) **a.** Stimuli are presented briefly (e.g., 100 ms) and are followed by a blank retention interval (e.g., 900 ms). **b.** One of the presented items is cued for recall (via the "?") and participants indicate their memory for the item's original color by clicking on the color circle.

Our illustration of Bayesian parameter estimation with multiple parameters uses the influential "mixture model" of visual working memory (Zhang and Luck, 2008). We illustrate the model and the estimation of its parameters with an extension of the Metropolis-Hastings algorithm already introduced.

#### The Mixture Model of Visual Working Memory

One popular visual working memory task involves the procedure shown in Figure 7.3. On any given trial, participants briefly view a display of a varying number of colored squares (typically in the range of 2–8; Panel **a**), and a short time later they recall the color of one of those squares, cued at random, with the aid of a color circle (Panel **b**). The stimuli are reproduced in grayscale in Figure 7.3; in reality the squares would be of different colors, and the color circle would, of course, be in color (with blue in the southwest, red in the north, and green in the southeast).

Typical results from a color-recall experiment are shown in Figure 7.4, which displays data for a single subject in the color estimation experiment reported by Zhang and Luck (2008) and made available by van den Berg et al. (2014). We show the proportion of responses for set sizes 3 and 6 at various distances (measured in degrees along the color circle) from the correct response: The vertical dashed line indicates 0 error and corresponds to the participant clicking on precisely the correct location on the color circle. Values to the left of that line translate into increasingly greater error in one direction (e.g., counterclockwise), and values to the right correspond to errors in the other direction (e.g., clockwise) along the color circle.

![](img/_page_177_Figure_2.jpeg)

**Figure 7.4** Data (circles) from a single subject in the color estimation experiment of Zhang and Luck (2008) and fits of the mixture model (solid lines). Responses are shown for set size 3 (open circles, dashed line) and set size 6 (filled circles, solid line). See text for details.

The figure shows that accuracy, unsurprisingly, declines as set size increases. That is, there are fewer filled circles near the vertical zero-error line than there are open circles. It is also apparent that small errors are more frequent than large errors. This is also unsurprising because even when memory is perfect, it is quite difficult to exactly match a memorized color from a wheel that offers a nearly infinite number of response options. A final intriguing aspect of the figure is that for the larger set size, once absolute errors exceed ±30◦ the proportion of errors does not decline further but remains flat – if anything, there is an upturn at the more extreme ends. There are a number of explanations for this attribute of the distribution of errors. A particularly simple and elegant explanation that we focus on here was proposed by Zhang and Luck (2008) in their mixture model.

According to the mixture model, people either remember an item by encoding it in one or several of a limited number of "slots" in memory, or they fail to remember it altogether. In the former case, they have information about the item available and their recall performance should be captured by some peaked and symmetric distribution that straddles the correct response. In the latter case, people would have no information about the item available at all, and performance should be effectively random with responses uniformly dispersed along the color circle. The merits of this "slot" idea have been extensively debated (Bays et al., 2011; Donkin et al., 2013; Kary et al., 2015; Ma et al., 2014; van den Berg et al., 2014), but here we side-step that debate and focus on the technical aspects of fitting this simple model to the data shown in Figure 7.4.

#### Fitting the Mixture Model via MCMC

Listing 7.3 presents a self-contained R function that performs MCMC parameter estimation. This function, called getMixtmodel, in turn calls a few auxiliary functions that are not shown in this listing but are explained later. Much of this function should be familiar by now, with the main difference being that it extends the MCMC sampling to a model with more than one parameter.

```
1 getMixtmodel <−function ( data , svalues) {
2 chain <− matrix (0 ,5000 ,2)
3 burnin<−500
4 set . seed(1234)
5 propsd <− svalues∗ .05
6 lb <− c (0 ,4)
7 ub <− c (1 ,360)
8
9 chain [1 ,] <− svalues # starting values for parameters
10 for (i in c (2: dim (chain) [1]) ) {
11 cur <− chain[i−1,]
12 doitagain <− TRUE
13 while (doitagain) {
14 propl <− cur + rnorm (2 ,0 ,propsd)
15 doitagain <− any (propl<lb) | | any (propl>ub)
16 }
17
18 lpropval <− logmixturepdf( data , propl [1] , propl [2])
19 +logprior(propl [1] , propl [2])
20 lcurval <− logmixturepdf( data , cur [1] , cur [2])
21 +logprior( cur [1] , cur [2])
22 llratio <− exp (lpropval−lcurval)
23 i f ( runif (1) < llratio) {
24 chain[i , ] <− propl
25 } else {
26 chain[i , ] <− cur
27 }
28 }
29 finparm<−apply (chain[−c (1: burnin) ,] ,2 , mean )
30 print (finparm)
31
32 td<−c ( −180:180)
33 pred<−(1−finparm [1]) ∗
34 dvonmises(mkcirc(td) ,mkcirc (0) ,sd2k(finparm [2]) )+
35 finparm [1] ∗ dunif (td, −180,180)
36 posterior<−chain[−c (1: burnin) ,]
37 return ( list (preds=pred , posteriors=posterior) )
38 }
```

**Listing 7.3** Function to conduct MCMC for the mixture model

Function getMixtmodel receives two input arguments (Line 1), either from the command line or from some other program that calls it. The argument data is a vector of numbers, expressed in degrees, that correspond to the error associated with each response of the participant. A histogram of data would yield a shape that is described by the open or filled circles in Figure 7.4. The other argument, svalues, contains the starting values of the two parameters of the mixture model, *g* and *σvM*, that will be used to initiate the MCMC sampling.

The first parameter, *g*, corresponds to the proportion of responses that reflect random guesses. In the mixture model, *g* captures the weighting of the probability densities for a uniform distribution covering the domain [−180, 180*)*. The second parameter, *σvM*, describes the standard deviation of a "circular" Gaussian distribution with mean zero to describe responses that are not guesses but reflect memory for the item. The mean of this distribution is zero to reflect the assumption that when people have a memory for an item, their representation is not systematically biased. Hence errors arising from imprecision in memory are assumed to be zero-centered. Because responses were distributed around the color circle used in the experiment (Figure 7.3), we cannot use a standard Gaussian distribution here. Rather, we use a distribution called the von Mises distribution, which can be thought of as a Gaussian distribution on a circle.

The starting values are first used to determine the standard deviation of the proposal distribution (Line 5). Because proportions are limited to the range 0–1 whereas the standard deviation of people's errors, expressed in degrees, can be far greater (Figure 7.4), the proposal distribution has to be tuned to each parameter separately. The tuning of the proposal distribution is immediately followed by specifying lower (lb) and upper (ub) bounds for the parameters. The lower bound for *σvM* is necessary for computational reasons because lower values can cause overflows or underflows (the lower bound of 4◦ is far lower than any value that one might expect to see in an experiment).

The MCMC iterations are contained within the loop that starts in Line 10. Little has changed from Listing 7.1 except that chain now has multiple columns, one for each parameter. Likewise, the proposal sampled in Line 14 returns two values rather than one. Note that the proposals for the two parameters are sampled independently. In principle, there is no reason why the proposals could not be correlated to each other if we had reason to assume a correlation between parameters.

A few other aspects of the MCMC loop deserve consideration. First, the sampling from the proposal distribution is embedded within a while loop (Line 13) that discards proposals that violate the range restrictions placed on the parameters. There is no point in seeking to evaluate the likelihood of the model at a point at which it is not defined (e.g., guesses exceeding 100%).

Second, when comparing the values of the target distribution for the current and proposed samples (in Lines 18 and 20), we use the logarithms of the likelihoods and priors. The use of logarithms guards against numerical issue that arise when very small (or large) numbers are multiplied. In consequence, multiplication turns into addition: Whereas we multiplied the prior and the likelihood in Listing 7.2, here we add the logarithms together. Likewise, the ratio of the two values for the target distribution is computed by subtraction rather than division (Line 22). Because that ratio must be in the range 0–1 to permit comparison against a random uniform number, the difference operation on logarithms is exponentiated in the same line in order to return to the original untransformed space. (The exponentiated difference between two logs is the same as the ratio between the original numbers).

Finally, we have increased the efficiency of the decision about whether or not to accept the proposal. In the previous example (Listing 7.2), we first asked whether the value of the target distribution was greater for the proposal than for the current sample. If it was, we accepted the proposal, and if it was not, we used a second test that compared the ratio of values to a random number to decide whether the proposal should nonetheless be accepted. Here, we combine those two decisions into a a single test in Line 23. This single test works because if the value for the proposal is greater than for the current sample, then the ratio of values is necessarily greater than one – and thus the random uniform number can never be greater than the ratio and thus the proposal will always be accepted. When the ratio is less than 1, the test in Line 23 works as it did in the preceding examples.

Once the MCMC chain has completed, the burnin period is discarded and the remaining samples are used to compute and then print the final parameter estimates, which are taken to be the mean of all posterior samples (Line 29). The function concludes by computing the predicted probability densities of the mixture model for the average of the sampled parameter values. It is those predictions, captured in pred, that are plotted as solid lines in Figure 7.4.

Function getMixtmodel shares much in common with the earlier example of MCMC and, at an abstract level, it should therefore be relatively easy to follow. However, we have thus far skipped over the core novelty of this example, namely the computation of the mixture model and the determination of priors. Those two important components are shown in Listing 7.4 together with a few other functions that are required for computation. As indicated in the listing, some of those functions were adapted from Suchow et al. (2013) and Bays et al. (2009).

```
1 library (circular)
3 # pdf f o r mixture model ( Suchow et al . , 2013)
4 logmixturepdf <− function ( data , g , sdv) {
5 data4vm <− mkcirc( data )
6 return (sum ( log ((1−g) ∗dvonmises(data4vm , mkcirc (0) , ←	
        sd2k(sdv) ) ←	
7 + g∗ dunif ( data , −180,180) ) ) )
8 }
9
10 # c o n v e r t SD i n t o Kappa ( Suchow e t a l . , 2013)
11 sd2k<−function (d) { #input is in degrees
12 S <− (d / 180) ∗pi #go to radian s
13 R <− exp(−Sˆ 2 / 2 )
14 K = 1 / (Rˆ 3 − 4∗Rˆ2 + 3∗R)
15 i f (R < 0.85) {K <− −0.4 + 1.39 ∗R + 0.43 / (1−R) }
16 i f (R < 0.53) {K <− 2∗R + Rˆ3 + (5 ∗Rˆ5) / 6}
17 return (K)
18 }
19
20 # j e f f r e y s p ri o r f o r p r e ci si o n ( Suchow et al . , 2013)
21 jp4kappa <− function (K) {
22 z <− exp ( ( log ( besselI (K ,1 , TRUE)) + K) −
23 ( log ( besselI (K ,0 , TRUE)) + K) )
24 return (z ∗ (K − z − K∗z ˆ2) )
25
```

```
26
27 #jeffreys prior for a pr oportion
28 jp4prop <− function (p) {pˆ −0.5 ∗ (1−p) ˆ −0.5}
29
30 # get ove rall p rio r for model parameters
31 logprior<− function (g , sdv) {
32 return ( log (jp4kappa(sd2k(sdv) ) )+ log (jp4prop(g)))
33 }
34
35 #make i t c i r c u l a r in deg rees
36 mkcirc<−function (td)
37 { a s . circular(td , control . circular= list (units="degrees" ) ) }
```

**Listing 7.4** Various functions for the mixture model

The functions rely on the R package circular, which is therefore loaded into R's workspace in the first line of the code. The core function is logmixturepdf which returns the likelihood *P(y*|*θ)* when provided with the data, *y*, and the parameter values *θ* (i.e., *g* and *σvM*) as parameters. The function consists of two statements: the data are first converted into the "circular" type using the mkcirc function (in Line 36 at the end of the listing). This is required because the dvonmises function in the next statement (Line 6), which is part of the circular package, expects input of this type.

Line 6 performs the bulk of the work by computing and then adding together the log-likelihoods for the von Mises distribution and the uniform distribution. The two likelihoods are weighted by the guessing parameter, *g*, which determines the fraction of guesses and hence multiplies the density of the uniform distribution. The responses that are not guesses are modeled by the von Mises distribution which is therefore weighted by 1−*g*. For the reasons noted earlier, the likelihoods are converted to logarithms before they are finally summed across all data points into a single value.

The dvonmises function is supplied with arguments that contain the data (converted into the required circular type), the mean of the distribution (which is zero, for the reasons noted earlier), and an estimate of the spread of that distribution. Unlike many other R functions (e.g., dnorm), for dvonmises the spread of the distribution is not provided as a standard deviation but as a precision. The precision is the reciprocal of the variance, and the conversion from standard deviation (and transformation from degrees to radians) is achieved in the sd2k function which commences in Line 10. The details of this function need not concern us here.

The next three functions, jp4kappa, jp4prop, and logprior provide the prior probabilities of specific parameter values, *P(θ)*, given some assumptions about their prior distribution. Here, we employ noninformative Jeffreys priors, which we already discussed in Section 6.3.1. Those priors are implemented in functions jp4kappa and jp4prop for the von Mises and uniform distribution, respectively (Suchow et al., 2013). One of the desirable properties of Jeffreys priors is that they seek to maximize the effect of the data evenly across the entire range of permissible values of the parameter in question.<sup>2</sup>

<sup>2</sup> Strictly speaking, this property holds only when a single parameter is estimated. With multiple parameters, the situation changes slightly as we will see in the next chapter. This nuance need not deter us from using Jeffreys priors in this example.

![](img/_page_182_Figure_2.jpeg)

**Figure 7.5** Posterior distributions of parameter estimates for *g* and *σvM* obtained when fitting the mixture model to the data in Figure 7.4. The top row of panels shows estimates for set size 3, and the bottom row for set size 6. In each panel, the thick vertical line represents the posterior mean and the thin dotted lines represent the 5th and 95th percentile of the posterior distribution. Note that all abscissas have different scales.

#### Predictions and Posterior Distributions

The predictions of the mixture model obtained with function getMixtmodel were already shown earlier in Figure 7.4. The corresponding posterior distributions of parameter values, also returned by logmixturepdf, are shown in Figure 7.5.

It appears that this particular participant rarely guessed at random (mean *g* = .01 for both set sizes), and that his or her precision of memory decreased considerably with setsize (mean *σvM* = 39.6 and *σvM* = 86.3). Unsurprisingly, those estimates for a single participant differ considerably from the values reported by Zhang and Luck (2008) for their overall sample.

# **7.2 Problems Associated with MCMC Sampling**

Any MCMC algorithm is subject to at least two problems. The first problem is known as the convergence problem, which means that the sampler is stuck in a subspace that it is unable to break out of. The second problem arises from the fact that MCMC involves a Markov process. It follows that even after the burn-in period, successive samples in the chain cannot be independent because, by definition, a Markov chain is based on a transition matrix that governs the transition from any state *i* to *i* + 1. Until now, we have ignored both of those problems because it allowed us to keep our R programs simple. Now we need to address both issues.

## 7.2.1 Convergence of MCMC Chains

Until now, our discussion of MCMC methods has tacitly assumed that the algorithm will map out the entire posterior distribution with sufficient accuracy if given enough time to do so. If a chain tours the entire distribution and if the exploration of the distribution is no longer dependent on starting values, then convergence is said to have occurred. However, in at least some circumstances, such convergence may remain elusive.

In particular, any multimodal posterior – for example, a distribution with two distinct peaks and a very shallow valley in between – may challenge MCMC methods (e.g., Justel and Pena, 1996; Matthews, 1993). Other cases in which MCMC algorithms may ˜ fail involve the detection of outliers in a regression situation. Those situations are usually modeled by estimating a mixture of distributions and estimating the posterior probability of each observation's membership in one or the other distribution (Justel and Pena, ˜ 1996). This can occur because the outlying data points exert considerable leverage on the regression line – that is, the line is unduly pulled toward those points – and once that has happened, the residuals are so small that the data points are unlikely to be reclassified as outliers.

The convergence problem has attracted much research attention. Although the preceding examples involve statistical estimation rather than cognitive modeling, the convergence problem transcends specific applications and we need to be aware of it whenever we apply MCMC techniques. At a conceptual level, the MCMC convergence problem differs little from the problems associated with the parameter-estimation procedures that we discussed in Chapter 3. Thus, there is no guaranteed solution to the convergence problem, although there are several safeguards that we can take, which again resemble those discussed in Chapter 3.

The most obvious solution, which is instantiated in virtually all MCMC statistical packages, including the JAGS package that we will introduce in the next chapter, is to perform and analyze multiple MCMC chains, each of which originates from different starting values. In a nutshell, just as with other parameter-estimation techniques, if all chains converge on the same result despite originating from a very different place, then one's confidence in the solution can be reasonably enhanced (although there are situations in which convergence may be more apparent than real; Matthews, 1993).

The use of multiple chains raises two issues: First, we need to decide on the best choice of starting values. Second, we need to interpret the results across our chains, and in particular, we need a criterion for their behavior that tells us when all is well and the chain(s) have converged on the posterior, and conversely, when we might face a problem.

Concerning the choice of starting values, there is considerable agreement that the values should be "overdispersed" – that is, they should be sufficiently far apart from each other to extend beyond the likely extent of the posterior distribution (e.g., Kass et al., 1998). Gelman and Rubin (1992) provide formal guidance on how to choose starting values. In the preceding example, our starting values were chosen to fall outside the likely centroid of the bivariate normal distribution, and further runs with different starting values would advisedly also put those starting points outside the likely centroid.

Turning to convergence diagnostics, this is an area of ongoing research (e.g., Cowles and Carlin, 1996), and multiple criteria exist and continue to be used. In R, the package superdiag provides an entire suite of convergence diagnostics. We will discuss an example of convergence diagnostics in the next chapter.

#### 7.2.2 Autocorrelation in MCMC Chains

We noted briefly at the beginning of this chapter that, by definition, successive samples in an MCMC chain must be correlated. This is because any Markov process is governed by a transition matrix (or "kernel" in the case of continuous variables) that constrains the number of possible pairwise sequences. Intuitively, this is obvious because each step involves a proposal that is sampled relative to the current position in parameter space. It is only when one considers samples that are spaced further apart along the chain that they become essentially independent. Given that we are interested in independent samples from a distribution, the autocorrelations among successive MCMC samples may appear problematic at first glance. Indeed, in an extreme case, if all successive samples were nearly identical to each other because of autocorrelations near unity, then it is easy to see that the chain would take a long time indeed to map out an entire distribution, compromising attempts to understand that distribution.

One solution to the autocorrelation problem is a process known as "thinning," which avoids dependence among samples by only considering every *k*th sample, where *k* may range from 2 to 40 or even greater (Link and Eaton, 2012). The computational expense of thinning is immediately apparent: if *k* = 40, then to obtain 10 usable samples, we need to generate 400 overall, of which we then discard 390.

Fortunately, it turns out that thinning is unnecessary in most instances, and we mention it here primarily to prepare you for an encounter with this term in the literature. As shown by MacEachern and Berliner (1994), when emphasis is on estimating the mean of a posterior distribution, nothing is gained by thinning. When the emphasis is on estimating variance – for example, to compute credible intervals for a parameter – thinning is still unlikely to be necessary (Link and Eaton, 2012; MacEachern and Berliner, 1994), although there are occasions when it is mandated because the autocorrelations in a chain are exceedingly high (Wabersich and Vandekerckhove, 2014).

#### 7.2.3 Outlook

We presented two detailed examples of the MCMC approach to Bayesian parameter estimation. In both cases, we presented R code to perform the MCMC via the Metropolis-Hastings algorithm and to implement the model under consideration. This approach is, however, limited in its applicability.

Listings 7.1 and 7.3 are sufficient to introduce MCMC but they fall short of providing the full computational power and convenience that is required for parameter estimation in realistic circumstances. The next chapter is therefore again devoted to Bayesian model fitting and model selection, but rather than using our own code to perform MCMC sampling, that chapter will make use of the JAGS software package. JAGS permits MCMC exploration of any model whose likelihood can be expressed in the JAGS language. As we will see, the JAGS language interfaces with R relatively seamlessly.

# **7.3 Approximate Bayesian Computation: A Likelihood-Free Method**

Up to now, we have assumed that the likelihood *P(y*|*θ)* can be computed (see Table 7.1). At first glance it might appear surprising and counterintuitive that there are situations in which one cannot compute the likelihood – after all, if I have a model and I know the parameters, how could I not know what its predictions are? As we will see very shortly, however, those situations arise rather frequently.

When the likelihood function cannot be computed, a relatively novel technique known as Approximate Bayesian Computation (ABC) can nonetheless permit us to apply Bayesian techniques to parameter estimation. (For a brief history of ABC, see Sunnaker et al., 2013). In a nutshell, ABC replaces computation of the likelihood ˚ function with a simulation of the model in question (Hartig et al., 2011). Thus, instead of computing the likelihood using a mathematical expression such as the single line of code in function logmixturepdf in Listing 7.4, we replace *P(y*|*θ)* with its simulated approximation, akin to the way in which we have already used MCMC to replace computation of the posterior distribution *P(θ*|*y)* with a sampled approximation.

## 7.3.1 Likelihoods That Cannot be Computed

Many models in cognition are not formulated in a way that permits computation of the likelihood. For example, the neural network models that we will discuss in Chapter 13 are generally not amenable to such computation. Other examples include cognitive architectures, such as ACT-R (e.g., Anderson, 1983a; Anderson and Lebiere, 1998; Anderson, 2007), which are complex models derived from research in artificial intelligence and which seek to explain a broad range of cognitive findings. (For an overview, see Lewandowsky and Oberauer, 2016).

Even relatively straightforward memory models, such as the TBRS\* (Oberauer and Lewandowsky, 2011), SOB-CS (Oberauer et al., 2012), REM (Shiffrin and Steyvers, 1997), and BCDMEM (Dennis and Humphreys, 2001) cannot be computed directly.3 Notwithstanding the absence of a directly computable solution, in all those cases the

<sup>3</sup> Numerical solutions for BCDMEM have become available since the model was originally published (Myung et al., 2007), but for a long time the model was considered to be uncomputable.

predictions of the model can be simulated with relative ease. To illustrate, items on a to-be-memorized list can be represented by randomly sampled feature values, and the success of encoding and subsequent retrieval of those items can be estimated by simulating many such items across multiple study-test trials (e.g., Oberauer et al., 2012).

For the remainder of the discussion, we therefore make the basic assumption that we have access to simulated results from our model under a given set of parameters. We describe the simulation by the notation *η(θ)*, and because the simulation is stochastic, different runs of the simulation under identical parameter values will give rise to a distribution of outcomes *X*, which we denote as *X* ∼ *η(θ)*.

#### 7.3.2 From Simulations to Estimates of the Posterior

#### An Abstract Overview of the Sampling Process

Figure 7.6 provides an overview of the simplest possible ABC algorithm, known as the rejection algorithm. The example involves a single hypothetical parameter, *θ*, whose prior distribution is shown at the top. We wish to obtain a posterior distribution for this parameter, *P(θ*|*y)*, based on the observed data, *y*, which are schematized in the gray shaded box at the top left.

For this example, we assume that the likelihood, *P(y*|*θ)*, is not computable. We instead resort to simulation, as shown in the middle row of panels. For each simulation *i*, we sample a particular value of *θ<sup>i</sup>* from its prior distribution and use that to generate a model outcome, *Xi*, via our simulation *η(θ)*. Each outcome is of the same form as the data in terms of number of observations and experimental conditions and so on. We perform many such simulations and then choose those that were "successful" to infer the posterior distribution of *θ*.

The success of simulation *i* is defined by how closely its result, *Xi*, resembles the data, *y*. To measure the proximity between *Xi* and *y*, both are first summarized by a single statistic – in this case *μ* for the data and *μ<sup>i</sup>* for the simulation. Note that the summary statistic *μ* is completely different from the parameter *θ*; for example, *μ* might be a sample mean expressed in milliseconds (for response times) whereas *θ* might be a probability, and so on. Once those summary statistics have been computed – and for the moment we will defer discussion of how and why this is done – we use some function *ρ(μi*,*μ)* to summarize the discrepancy between simulation *i* and the data. (See Chapter 3 for discussion of discrepancy functions.) A common discrepancy measure is the sum of squared error between some summary statistic of the simulated and observed data (Turner and Van Zandt, 2012). Whenever that discrepancy falls below some small number , we accept simulation *i* as a plausible candidate, and we add the associated value of *θ<sup>i</sup>* to our posterior sample.

Over many such trials, we obtain a sampled distribution *P(θ*|*ρ(μi*,*μ) < )*. If is sufficiently small, this sampled distribution will approximate the desired posterior distribution *P(θ*|*y)*, as illustrated at the bottom of Figure 7.6. In the limiting case, as → 0, the sampled distribution will be the exact posterior distribution. Conversely, as → ∞, the sampled distribution will exactly match the prior.

![](img/_page_187_Figure_2.jpeg)

**Figure 7.6** Overview of a simple Approximate Bayesian Computation (ABC) rejection algorithm. See text for details. Figure reprinted with permission from Sunnaker et al. (2013). ˚

We now take up all those steps involved in ABC in more detail and provide an illustrative example.

#### Sufficient Summaries for the Data and Simulation Results

At the heart of ABC is the function *ρ(*·*)* that computes the discrepancy between the observed data *y* and a simulation result *X*. For computational ease, the function *ρ(*·*)* operates on summary statistics (Turner and Van Zandt, 2012), such as the mean of the data and predictions. In Figure 7.6, we used the mean *μ* to summarize data and predictions. Because the mean is only one of a number of possible summary statistics, we refer to the collective of possible statistics by *S*.

The choice of summary statistic is nontrivial (Turner and Van Zandt, 2012). Ideally, the statistic should be *sufficient* – that is, *S* should contain as much information about the parameter *θ* as the entire data set. For example, the sample mean is a sufficient statistic for a normal distribution with unknown mean but known variance. Once we have computed the mean (or a vector of means if we have multiple experimental conditions or participants), no further information can be obtained from the sample that would improve our knowledge about the distribution parameter. By contrast, for an arbitrary distribution the sample *median* is not sufficient to estimate the population *mean*. Even after we have computed the median, the sample could provide further information about the mean – for example, if it contained a large number of positive outliers, the mean would be greater than the median, and this information embedded in the sample was lost during computation of the median. Establishing the sufficiency of a statistic is not entirely trivial and we do not concern ourselves with the details here; interested readers are referred to Turner and Van Zandt (2012).

Sufficiency is only half the story. The other half involves the choice of *ρ(*·*)*. Suppose we have settled on sample means for our summary statistic, as in Figure 7.6. We still have to make choices about *ρ(*·*)*. For example, should we use *ρ(μi*,*μ)* = |*μ* − *μi*|? Or should we use *ρ(μi*,*μ)* = *(μ* − *μi)*2? This choice is also nontrivial because in the absence of knowledge of the likelihood function – remember that we would not be conducting ABC if that function were computable – we are stumbling somewhat in the dark with respect to sufficiency of *S* as well as to the choice of *ρ(*·*)*. Fortunately, for many models those choices need not materially affect the outcome (Turner and Van Zandt, 2012), and we can confidently proceed to discuss our example, before we take up the various nuanced risks that are relevant to ABC in a concluding section.

#### 7.3.3 An Example: ABC in Action

For our example, we consider a recognition memory experiment in which participants first study a list of words and are then presented with a long sequence of test items. Each test item is either a word from the study list (old) or an item that has not been seen before (new); old and new items usually appear with equal probability in the test sequence. The participant responds to each test item by indicating whether it is "old" or "new." Hypothetical data from such an experiment are shown in Figure 7.7 **a**. In this instance, the participant scored a "hit" (recognizing an old item correctly) 60% of the time, and the person committed "false alarms" (incorrectly flagging a new item as old) 11% of the time. The person correctly rejected new items 89% of the time, but they also missed old items 40% of the time.

#### Describing Recognition Memory Performance with a Signal-Detection Model

Figure 7.7 **b** presents a signal-detection model of that participant's performance. Recognition performance is frequently analyzed with a variety of signal-detection models (e.g., Jang et al., 2009; Wixted, 2007). Here we focus on the simplest possible equal-variance model (Swets et al., 1961). The model assumes that memory is captured by a unidimensional, noisy familiarity signal. For new items, the familiarity is centered around zero (i.e., no memory on average), as shown by the left-hand Gaussian

![](img/_page_189_Figure_2.jpeg)

**Figure 7.7 a.** Data from an hypothetical recognition memory experiment in which people respond "old" or "new" to test items that are old or new. **b.** Signal-detection model of the data in panel **a**. Shading of the areas corresponds to shading of the cell entries in panel **a**. The two parameters of the model, *d* and *b*, are identified. The vertical dotted line represents the optimal criterion and the solid vertical line represents the participant's actual criterion placement as determined by parameter *b*. The dark gray area under the new-item curve represents false alarms and the light area under the old-item curve represents hits.

distribution in Figure 7.7 **b**. For old items, by contrast, the distribution is shifted upward to represent the fact that study of the items has raised their familiarity. This is represented by the right-hand distribution, which is shifted upward by an amount captured by the parameter *d*. The magnitude of *d* is determined by experimental variables such as presentation duration, retention interval, and so on.

People are presumed to convert this noisy familiarity signal into overt responses via a decision criterion. Any perceived familiarity above the criterion would yield an "old" response, whereas any familiarity below generates a "new" response. The ideal observer would place the criterion half-way between the two distributions (i.e., at *d/*2), indicated by the dotted vertical line in Figure 7.7 **b**. (This assumes that signal and noise trials have equal probability, as we do here throughout; see Knoblauch and Maloney, 2012, for a detailed derivation.) Even with an optimally placed criterion, errors would be unavoidable because the two distributions overlap considerably. In the example depicted in Figure 7.7 **b**, the participant used a more conservative criterion: items need to be substantially familiar before the person is willing to say "old." This is indicated by the solid vertical line in the figure. In our parameterization of the model, the criterion placement is represented by the parameter *b*, which expresses the offset of the actual response criterion relative to the optimal placement (Stanislaw and Todorov, 1999). Positive values, as in this case, move the criterion to the right and represent increasing conservatism – that is, a tendency to reduce false alarms at the expense of more misses. Negative values would represent a more liberal placement of the criterion – that is, a tendency to minimize hits at the expense of increased false alarms. To illustrate this decision process, two of the areas under the curves in Figure 7.7 **b** are identified by shading. The corresponding cells in Figure 7.7 **a** are shaded in the same manner. The areas under the curve to the right of the criterion directly give us the probabilities in the table. The probability of a hit is the probability of a sample from the signal distribution exceeding the criterion, which is in turn given by the complement of the cumulative probability distribution for the normal density (i.e., how much of the signal distribution is above the criterion?). Thus, on trials when a new item is presented, the observer will nonetheless classify it as "old" 11% of the time (dark shading). When an old item is presented, the observer will correctly classify it as "old" 60% of the time (light shading).

For this illustration, the parameter values are *d* = 1.5 and *b* = 0.5. Those values are, of course, readily computable by looking up of the various areas under the normal distribution. However, for this example we will ignore this convenient shortcut and will instead resort to ABC. This approach is implemented in Listing 7.5.

```
1 y <− c (60 ,11) #define target data
2 dmu <− 1 #define hyperparameters
3 bmu <− 0
4 dsigma <− bsigma <− 1
5
6 ntrials <− 100
7 epsilon <− 1
8 posterior <− matrix (0 ,1000 ,2)
9 for (s in c (1:1000) ) { #commence ABC
10 while (TRUE) {
11 dprop <− rnorm (1 ,dmu , dsigma)
12 bprop <− rnorm (1 ,bmu , bsigma)
13 X<−simsdt(dprop , bprop , ntrials) #simulate proposal
14 i f ( sqrt (sum ( ( y−X) ˆ2) ) <= epsilon) { break }
15 }
16 posterior[s , ]<−c (dprop , bprop) #keep good simulation
17 print (s) #show sign of l i f e
18 }
```

**Listing 7.5** ABC of a signal-detection model

#### An ABC Approach to Signal-Detection Modeling

The program begins (Line 1) by defining the target data, *y*, based on the results in Figure 7.7 **a**. Our objective is to estimate the values of *d* and *b* that are associated with those results. Note that we only provide hits and false alarms because the other response classes (misses and correction rejections) can be derived by subtraction.

The subsequent lines define the hyperparameters for the prior distributions of *d* and *b*. We assume on the basis of prior knowledge that the mean of those distributions is 1 and 0, respectively, and their standard deviation 1. The mean of 1 for *d* is based on the vast body of existing research showing that people tend to remember items on a study list (of course they do!), and the mean of 0 for *b* is based on the assumption that, on average, people are unbiased in their criterion. The assumption that *d* and *b* are normally distributed is not coincidental. The hit rates and false alarm rates are related to the values of *d* and *b* via the cumulative distribution function (CDF) of the normal distribution, which defines the areas under the curves shown in Figure 7.7 **b**. One fundamental result in statistics is the probability integral transform (Angus, 1994), which states that if any random variable *X* has a continuous CDF *F(*·*)*, then the random variable *Y* = *F(X)* is uniformly distributed. Because the normal CDF is continuous, the hit and false alarm rates that are associated with *d* and *b* are themselves uniformly distributed (see also Lee and Wagenmakers 2013, p. 158). Our assumed prior distributions are thus noninformative with respect to the data we might observe.

We then define the number of experimental trials (Line 6) and our acceptance criterion epsilon. We set epsilon to 1, which means that we will accept any simulation that yields data within ±1% of the observed percentage of hits and false alarms.

The core of the ABC algorithm commences in Line 9. Within the for loop sits a while loop whose exit condition is TRUE. Of course, this loop can never complete on its own and will continue forever unless the program exits by issuing a break command. This happens in Line 14, conditional on the simulated results, X, being close enough (i.e., ≤epsilon) to the data, y. We use a Euclidean distance measure to determine whether our simulated results are close enough to the data (Turner and Van Zandt, 2014). The Euclidean distance is effectively the same as the RMSD that we introduced in Chapter 3.

So where do the simulated data come from? The answer is provided by the three lines commencing in Line 11. We first obtain proposed values for the parameters *d* and *b* from their respective prior distributions. Note that we assume both parameters to be normally distributed with the hyperparameters defined as discussed earlier. We then submit those candidate values plus some information about the experimental design to the function simsdt, which returns the simulated candidate results X.

Although we know nothing about simsdt at this point, it should be clear that the algorithm in Listing 7.5 instantiates the procedure summarized in Figure 7.6: We sample candidate parameter values, simulated the associated results, and the parameter values of any simulation that is "good enough" are retained in our growing sample of the posterior distribution. Once the program completes, we just need to compute the means of the two columns in posterior to get our estimates of *d* and *b*. We obtained 1.45 and 0.48 when we ran this program, but you may obtain slightly different results owing to differences in randomization. However, the obtained values should be close to 1.5 and 0.5, which are the "true" parameter values that can be obtained by direct computation for the data in Figure 7.7 **a**. When we ran the script, the 2.5th and 97.5th quantiles of the posterior distribution were 0.90 and 2.00 for *d*, and 0.22 and 0.77 for *b*, respectively.

What is left is to have a closer look at function simsdt, which is shown in Listing 7.6. The function draws two lots of samples from the normal distributions in Figure 7.7 **b**, one for the old items whose means is the proposed value of *d*, and one for the new items with mean 0. Given the design of the experiment, with an equal number of old and new test items, we sample ntrials/2 events of each class.

The sampled familiarity values are then turned into responses by comparing them to the response criterion, which by the parameterization defined above is located at d/2+b (see the solid vertical line in Figure 7.7 **b**). Depending on whether the event was an old or new item, familiarity values that exceed the criterion are either classified as hits or false alarms, respectively, and those two values are then returned as percentages.

```
1 # simulate sdt given parameters and number of t r i a l s
2 simsdt<− function (d , b , ntrials) {
3 old <− rnorm (ntrials / 2 ,d)
4 hits <−sum ( old >(d / 2+b) ) / (ntrials / 2 ) ∗100
5 new <− rnorm (ntrials / 2 ,0)
6 fas <− sum ( new>(d / 2+b) ) / (ntrials / 2 ) ∗100
7 return (X<−c (hits , fas) )
8 }
```

**Listing 7.6** Simulating a signal-detection model for ABC

#### ABC in Practice

This example served to illustrate the ABC approach but it has at least one limitation that prevents its direct applicability in practice. The algorithm in Listing 7.6, known as rejection sampling, is extremely wasteful. That is, it may take many iterations of the while loop before a simulation is close enough to the data to be accepted. This matters little in this instance, but with more complex models and larger data sets, rejection sampling is astronomically inefficient and basically no longer of much use. In reality, ABC therefore nearly always relies on more sophisticated sampling algorithms. To illustrate briefly, Turner and Sederberg (2012) introduced a "differential evolution" version of ABC, known as ABCDE, We do not have space to discuss those more sophisticated algorithms in detail. Interested readers are referred to a review of various algorithms by Wilkinson (2013).

In practice, it is usually difficult and time-consuming to perform ABC by writing customized programs in R. Fortunately, there are two packages available to perform ABC: they are called *abc* and *EasyABC*, and are available from CRAN.

#### **7.4** *In Vivo*

#### MCMC: A Clever Way to Run the Numbers

*Don van Ravenzwaaij (University of Groningen)*

I must have first heard about Markov Chain Monte Carlo (MCMC) sampling in my undergraduate years. For the longest time, I struggled with understanding, *really* understanding, what the point was of MCMC sampling. The technique was usually explained something along the lines of this: "We would like to know what some unknown underlying probability distribution is, but we cannot observe it: we do not have an analytical expression to obtain said distribution. MCMC allows us to *approximate* this distribution by sampling. . . " And so on and so on; the rest did not much to contribute to my understanding. Sampling how? Surely, if we have a means to calculate probabilities based on the data, what do we need MCMC for? And if we do not have a means to calculate probabilities based on the data, what exactly does MCMC add?

My breakthrough came in one of the yearly Erasmus-IP (Intensive Program) Seminars on Mathematical Psychology, the 2009 edition in Tubingen. I had one year as a PhD ¨ student behind me, and had started to learn more about MCMC, but had still not tackled my original issue: what is this magic that MCMC contributes? One of the seminars in the 2009's edition of this program was given by Francis Tuerlinckx on MCMC sampling. There was a lot of excellent content in the seminar, but it was a single sentence Francis said that would have a profound influence on the rest of my career (and, incidentally, my understanding of MCMC sampling). Francis had been talking for a long time and my brain was bulging with equations, and here he goes: "Well, time to wrap up for me, it is almost lunch time. You could always, you know, for fun, try and build one of these Gibbs samplers yourself." Credits, the end, everyone off to the lunch hall.

Except I was there first. I front-loaded as much food as I could carry and went off to my dorm room with my trusty laptop and a precariously balanced pile of chow. I had an hour and a half to make this work and I'd be damned if I would not have a serviceable version of a Gibbs sampler before the seminar would resume. Lots of cursing could be heard in the adjacent rooms in the next hour and a half and I shall not claim my final product was particularly elegant, but work it did. After the lunch break I walked up to Francis and showed him my work. He smiled at me (a clear indication of how nice a person he is, as my code really was a mess), and said, "Good job."

So what exactly did I learn from building my own MCMC sampler? Well, for one, it is not a magic calculator. Armed with a likelihood, you can calculate probabilities for every conceivable combination of parameter values without needing an MCMC sampler (well, rounded to a certain decimal). The trick, as it turned out, is that doing all these calculations takes an incredible amount of time. The more parameters you are working with, the more combinations of parameter values there are, and the time it takes for all the calculations that are involved increases exponentially. What MCMC did was provide a clever way of figuring out what probabilities to calculate and approximate the underlying distribution based on a few strategically chosen calculated probabilities, rather than slogging through the whole grid of possible numbers.

The overarching thing I learned from this of course was: staring at a bunch of equations (though definitely useful) only gets you so far. By implementing "mathsy" stuff yourself, such as MCMC samplers, you get a much more thorough understanding of what is really going on under the hood. Over the course of the years, my understanding of MCMC samplers deepened, enough to write a tutorial about how they work and hopefully contribute to someone else's understanding of MCMC samplers (van Ravenzwaaij et al., in press).

The take-home message should be: "Do you want to know how something works? Build it yourself!" The examples provided in this chapter and in van Ravenzwaaij et al. (in press) should get you started, but don't shy away from experimenting with more complicated versions of MCMC samplers! Even if you never end up using them (I certainly did not touch my first Gibbs sampler ever again), the learning experience it provides could prove much more valuable.
