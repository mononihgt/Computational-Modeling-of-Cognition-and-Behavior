# 第 11 章：使用贝叶斯因子进行贝叶斯模型比较 (Bayesian Model Comparison Using Bayes Factors)

在上一章中，我们讨论了模型复杂性以及如何使用 AIC 等指标在拟合优度和复杂性之间进行权衡。本章我们将深入探讨**贝叶斯模型比较**的核心——**贝叶斯因子 (Bayes Factor)**。

贝叶斯方法通过**边缘似然 (Marginal Likelihood)** 自然地体现了奥卡姆剃刀原理（Occam's Razor），即在能够解释数据的前提下，优先选择更简单的模型。

## 1. 理论背景 (Theory)

### 1.1 边缘似然与奥卡姆剃刀
在贝叶斯参数估计中，我们通常关注后验分布 $P(\theta|y)$，而忽略分母 $P(y)$。但在模型比较中，这个分母——**边缘似然**（也称为**证据 Evidence**）——至关重要。

$$
P(y|M) = \int P(y|\theta, M)P(\theta|M) d\theta
$$

边缘似然不仅仅是最大似然（最佳拟合），它是模型在整个参数空间上的**平均拟合**。

*   **复杂模型**：参数空间大，能预测各种各样的数据。因为概率总和为 1，如果它把概率分散到各种可能的数据上，那么对于任何**特定**数据的预测概率就会降低（摊薄了）。
*   **简单模型**：参数空间小，只能预测有限的数据模式。如果观测数据恰好落在其预测范围内，它的边缘似然会很高。

这就是贝叶斯方法自动惩罚复杂模型的原因，无需像 AIC 那样人为添加惩罚项。

### 1.2 贝叶斯因子 (Bayes Factor)
贝叶斯因子是两个模型边缘似然的比值，用于衡量数据支持一个模型胜过另一个模型的程度：

$$
BF_{12} = \frac{P(y|M_1)}{P(y|M_2)}
$$

*   $BF_{12} > 1$：数据支持模型 1。
*   $BF_{12} < 1$：数据支持模型 2。
*   通常，$BF > 3$ 被认为是实质性证据，$BF > 10$ 是强有力证据。

## 2. 模型形式化 (Formalization)

### 2.1 积分问题
计算边缘似然的核心难点在于计算上述的积分。对于大多数心理学模型，这个积分没有解析解。本章介绍了三种主要的数值计算方法：

1.  **数值积分 (Numerical Integration)**：适用于低维参数空间（参数少于 10 个）。
2.  **重要性采样 (Importance Sampling)**：适用于高维参数空间，通过从一个易于采样的分布中采样来估计积分。
3.  **Savage-Dickey 密度比 (Savage-Dickey Density Ratio)**：专门用于**嵌套模型**比较（例如测试某个参数是否等于 0）。

## 3. 代码实现 (Implementation)

我们将通过三个案例来演示这三种方法。

### 3.1 方法一：数值积分 (Numerical Integration)
**案例**：比较遗忘的指数模型 (Exponential) 和幂函数模型 (Power)。
**代码文件**：`codeFromBook/Chapter11/numericalInt.R`

我们使用 R 的 `cubature` 包进行自适应数值积分。

```r
library(cubature)

# 定义指数模型的似然函数
expL <- function(theta,tlags,y,n){
  a <- theta[1]; b <- theta[2]; alpha <- theta[3]
  # 二项分布似然
  p <- dbinom(y,n,a+(1-a)*b*exp(-alpha*tlags))
  return(prod(p))
}

# 定义幂函数模型的似然函数
powL <- function(theta,tlags,y,n){
  a <- theta[1]; b <- theta[2]; beta <- theta[3]
  p <- dbinom(y,n,a+(1-a)*b*((tlags+1)^(-beta)))
  return(prod(p))
}

# 计算边缘似然 (Marginal Likelihood)
# adaptIntegrate 自动在参数空间 [0,0,0] 到 [0.2,1,1] 之间积分
expML <- adaptIntegrate(expL,c(0,0,0),c(0.2,1,1),
               tlags=tlags,y=nrecalled,n=nitems)
powML <- adaptIntegrate(powL,c(0,0,0),c(0.2,1,1),
                        tlags=tlags,y=nrecalled,n=nitems)

# 计算贝叶斯因子
BF = expML$integral / powML$integral
```

### 3.2 方法二：重要性采样 (Importance Sampling)
**案例**：比较信号检测论 (SDT) 和高阈限理论 (1HT)。
**代码文件**：`codeFromBook/Chapter11/SDT_Importance.R`

当参数较多时，数值积分变得困难。我们可以使用蒙特卡洛积分。为了提高效率，我们使用**重要性采样**。核心思想是从一个与后验分布相似的分布 $g(\theta)$ 中采样，以确保采样点集中在似然函数的高值区域。

在这里，构建了一个混合分布 $g(\theta)$，由先验分布和后验分布的近似（正态分布）混合而成。

```r
# ... (前文先通过 MCMC 获取了后验样本，并计算了均值 d_mu 和标准差 d_sd) ...

# 1. 构建重要性采样分布 g 的样本
# 混合比例 gmix = 0.2 (20% 来自先验，80% 来自后验近似)
mask <- runif(N) > gmix
d[mask] <- d_pos[mask] # d_pos 是从后验近似(正态分布)中采样的
B[mask] <- B_pos[mask]

# 2. 计算权重 pp = p(theta) / g(theta)
# 分子是先验，分母是混合分布 g
pp <- dnorm(d,1,1) * dnorm(B,0,1) /
  ((1-gmix)*dnorm(d,d_mu,d_sd)*dnorm(B,B_mu,B_sd) + gmix*df(d)*dB(B))

# 3. 计算加权似然的均值 -> 边缘似然
L <- dbinom(h,sigtrials,pnorm(d/2-B)) * 
     dbinom(f,noistrials,pnorm(-d/2-B)) * pp
ml_SDT <- mean(L)
```
对 1HT 模型重复此过程，然后计算比值 `ml_SDT / ml_HT`。

### 3.3 方法三：Savage-Dickey 密度比
**案例**：测试 SDT 模型中的偏差参数 $b$ 是否显著异于 0（即是否存在偏差）。
**代码文件**：`codeFromBook/Chapter11/SDT_Savage.R`

这是一个嵌套模型比较问题：
*   模型 $H_1$ (General)：$b$ 自由变化。
*   模型 $H_0$ (Null)：$b = 0$。

Savage-Dickey 密度比告诉我们，贝叶斯因子 $BF_{01}$ 等于**后验密度在 $b=0$ 处的值**除以**先验密度在 $b=0$ 处的值**。

$$
BF_{01} = \frac{P(b=0|y, H_1)}{P(b=0|H_1)}
$$

如果后验分布在 0 处比先验分布更高，说明数据增加了我们对 $b=0$ 的确信度，支持零假设。

```r
library(logspline)

# 获取 b 的后验样本
source("SDT_small.R")
mcmcs <- as.matrix(mcmcfin)

# 使用 logspline 估计后验密度函数
blogspl <- logspline(mcmcs[,"b"])

# 计算 Savage-Dickey 比率
# 分子：后验在 0 处的密度 dlogspline(0, blogspl)
# 分母：先验在 0 处的密度 dnorm(0, 0, 1)
BF <- dlogspline(0,blogspl) / dnorm(0,0,1)
```

## 4. 运行结果与讨论

*   **数值积分结果**：代码会输出一个具体的 BF 值。由于数据是模拟生成的（假设真实模型是指数模型），我们期望 $BF > 1$，支持指数模型。
*   **重要性采样结果**：SDT 和 1HT 都能很好地拟合 ROC 曲线上的点，因此 BF 通常接近 1，表明数据没有提供强有力的证据区分这两个模型（除非先验有很强的信息）。
*   **Savage-Dickey 结果**：通过比较 $b=0$ 处的后验和先验密度，我们可以直观地判断是否存在偏差。如果 $BF_{01} > 1$，说明支持无偏差假设。

### 总结
贝叶斯因子提供了一种连贯的框架来比较模型，它自动平衡了拟合优度和复杂性。虽然计算边缘似然具有挑战性，但通过数值积分、重要性采样和 Savage-Dickey 比率等方法，我们可以有效地解决这一问题。
