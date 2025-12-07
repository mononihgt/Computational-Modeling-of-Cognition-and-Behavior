# 第 8 章：使用 JAGS 进行贝叶斯参数估计 (Bayesian Parameter Estimation with JAGS)

本章我们将从理论转向实践，介绍如何使用 **JAGS (Just Another Gibbs Sampler)** 这一强大的工具来实现贝叶斯模型。JAGS 是一种专门用于分析贝叶斯层次模型的程序，它使用 **Gibbs 采样 (Gibbs Sampling)** 算法从复杂的后验分布中进行采样。

## 1. Gibbs 采样 (Gibbs Sampling)

### 1.1 理论背景 (Theory)

Gibbs 采样是马尔可夫链蒙特卡洛 (MCMC) 方法的一种特例。与 Metropolis-Hastings 算法不同，Gibbs 采样不需要设计“提议分布” (proposal distribution) 和计算接受率。

它的核心思想是：虽然我们很难直接从多维联合后验分布 $f(x, y)$ 中采样，但在很多情况下，我们可以很容易地从**条件分布** $f(x|y)$ 和 $f(y|x)$ 中采样。

Gibbs 采样通过交替更新每个参数来实现：
1.  固定 $y$，从 $f(x|y)$ 中采样得到新的 $x$。
2.  固定 $x$ (使用刚更新的值)，从 $f(y|x)$ 中采样得到新的 $y$。
3.  重复上述过程。

### 1.2 双变量正态分布示例 (Bivariate Normal Example)

假设我们要估计两个相关变量 $x$ 和 $y$ 的分布。

**形式化 (Formalization)**：

条件分布如下：
$$x^{(i+1)} \sim N\left(\rho\frac{\sigma_x}{\sigma_y}y^{(i)}, \sqrt{\sigma_x^2(1-\rho^2)}\right)$$
$$y^{(i+1)} \sim N\left(\rho\frac{\sigma_y}{\sigma_x}x^{(i+1)}, \sqrt{\sigma_y^2(1-\rho^2)}\right)$$

**代码实现 (Implementation)**：

文件：`codeFromBook/Chapter8/gibbs.R`

```r
# gibbs sampling
# 初始化参数
sxt1mr <- sqrt(sigx^2*(1-rho^2)) 
syt1mr <- sqrt(sigy^2*(1-rho^2))
rxy <- rho*(sigx/sigy)
ryx <- rho*(sigy/sigx)             
xsamp <- ysamp <- rep(0,nsamples)
xsamp[1] <- -2   # 初始值
ysamp[1] <- 2   

# Gibbs 采样循环
for (i in c(1:(nsamples-1))) { 
    # 基于当前的 y 更新 x
    xsamp[i+1] <- rnorm(1, mean=rxy*ysamp[i], sd=sxt1mr)  
    # 基于刚刚更新的 x 更新 y
    ysamp[i+1] <- rnorm(1, mean=ryx*xsamp[i+1], sd=syt1mr)  
}
```

这段代码展示了 Gibbs 采样的核心逻辑：在 `for` 循环中，`xsamp` 的更新依赖于上一轮的 `ysamp`，而 `ysamp` 的更新则立即使用了本轮最新的 `xsamp`。

## 2. JAGS 基础入门

JAGS 是一个独立的程序，我们需要通过 R 包 `rjags` 与其交互。使用 JAGS 通常涉及两个文件：
1.  **R 脚本 (.R)**：负责准备数据、调用 JAGS、运行采样、处理结果。
2.  **JAGS 模型文件 (.j 或 .txt)**：使用 JAGS 语言定义模型结构（先验和似然）。

### 2.1 简单正态分布估计

**模型定义 (JAGS)**：

文件：`codeFromBook/Chapter8/mymodel.j`

```plaintext
model {
    # 似然函数 (Likelihood)
	for (i in 1:N) { 
		xx[i] ~ dnorm(mu, tau)
	}
	# 先验分布 (Priors)
	mu ~ dunif(-100,100)  
	# 注意：JAGS 使用精度 (precision) tau = 1/sigma^2，而不是标准差
	tau <- pow(sigma, -2) 
	sigma ~ dunif(0, 100)
}
```

**注意**：JAGS 是**声明式 (Declarative)** 语言，语句的顺序不影响执行。`dnorm` 函数的第二个参数是**精度 (precision)** $\tau$，即方差的倒数 ($\tau = 1/\sigma^2$)，这与 R 中使用标准差不同，需要特别注意。

**R 脚本控制**：

文件：`codeFromBook/Chapter8/myscript.R`

```r
require(rjags)

N <- 1000
x <- rnorm(N, 0, 2)     

# 初始化模型
myj <- jags.model("mymodel.j",  
                   data = list("xx" = x, "N" = N)) # 将 R 变量映射到 JAGS 变量
update(myj, n.iter=1000) # Burn-in (预烧期)
mcmcfin <- coda.samples(myj, c("mu", "tau"), 5000) # 采样

summary(mcmcfin)
plot(mcmcfin)
```

## 3. 信号检测论 (Signal Detection Theory, SDT)

我们将第 7 章介绍的 SDT 模型用 JAGS 实现。

### 3.1 模型形式化

我们需要估计辨别力 $d'$ (代码中为 `d`) 和 偏好 $c$ (代码中为 `b`)。
命中率 (Hit Rate) $\phi_h$ 和 虚报率 (False Alarm Rate) $\phi_f$ 由下式给出：

$$\phi_h = \Phi(d/2 - b)$$
$$\phi_f = \Phi(-d/2 - b)$$

其中 $\Phi$ 是标准正态累积分布函数。

### 3.2 代码实现

**JAGS 模型** (`codeFromBook/Chapter8/SDT.j`)：

```plaintext
model{
	# 先验分布
    d ~ dnorm(1,1) 
    b ~ dnorm(0,1)
	
    # 转换为曲线下面积 (概率)
    phih <- phi(d/2-b)   # phi() 是 JAGS 的标准正态 CDF 函数
    phif <- phi(-d/2-b)
	
    # 观测数据 (二项分布)
    h ~ dbin(phih, sigtrials)  
    f ~ dbin(phif, noistrials)
}
```

**R 脚本** (`codeFromBook/Chapter8/SDT.R`)：

```r
# ... 数据准备 ...
# 初始化多条链 (Chains)
oneinit <- list(d=0, b=0)   
myinits <- list(oneinit)[rep(1,4)]
# 为每条链添加随机扰动，以测试收敛性
myinits <- lapply(myinits,FUN=function(x) lapply(x, FUN=function(y) y+rnorm(1,0,.1)))

sdtj <- jags.model("SDT.j", 
                   data = list("h"=h, "f"=f, 
                               "sigtrials"=sigtrials,"noistrials"=noistrials),
                   inits=myinits,
                   n.chains=4)  # 运行 4 条链
# ... Burn-in 和采样 ...
gelman.plot(mcmcfin) # Gelman-Rubin 收敛诊断
```

这里我们运行了 4 条独立的 MCMC 链，并使用 `gelman.plot` 来检查它们是否收敛到同一个分布（Gelman-Rubin 统计量应接近 1）。

## 4. 多项式处理树模型 (Multinomial Processing Tree, MPT)

MPT 模型常用于分析分类数据，假设观察到的反应是由一系列潜在的认知状态（如“记住”或“猜测”）决定的。

### 4.1 单高阈值模型 (One-High-Threshold Model, 1HT)

**理论背景**：
1HT 模型假设识别记忆有两种状态：
1.  **确定识别 (Detect)**：以概率 $\theta_1$ 进入该状态，此时肯定回答“旧的”。
2.  **不确定 (Uncertain)**：以概率 $1-\theta_1$ 进入该状态，此时以概率 $\theta_2$ 猜测“旧的”。

**形式化**：
$$P(\text{Hit}) = \theta_1 + (1 - \theta_1)\theta_2$$
$$P(\text{False Alarm}) = \theta_2$$

**代码实现** (`codeFromBook/Chapter8/1HT.j`)：

```plaintext
model{
    # Beta 分布作为概率参数的先验 (范围 0-1)
    th1 ~ dbeta(1,1) 
    th2 ~ dbeta(1,1) 
	
    # 预测反应概率
    predh <- th1+(1-th1)*th2    
    predf <- th2				
	
    # 似然函数
    h ~ dbin(predh, sigtrials)
    f ~ dbin(predf, noistrials)
}
```

### 4.2 Wagenaar & Boer (1987) 目击者记忆模型

这是一个更复杂的 MPT 模型，用于解释目击者面对误导信息时的记忆表现。

**理论背景**：
实验包含三个阶段：
1.  观看幻灯片（含交通灯）。
2.  接受提问（包含一致、不一致或中性信息）。
3.  再认测试（交通灯 vs 停车牌）。
4.  回忆颜色。

**无冲突模型 (No-Conflict Model)** 假设误导信息不会破坏原始记忆，只是在原始记忆提取失败时作为补充。

**代码实现** (`codeFromBook/Chapter8/wagenaar.j`)：

这个模型展示了如何在 JAGS 中处理多维数据和复杂的概率计算。

```plaintext
model { 
    # 先验
    p ~ dbeta(1,1) # 编码原始信息的概率
    q ~ dbeta(1,1) # 编码误导信息的概率
    c ~ dbeta(1,1) # 编码颜色的概率

    # 似然函数：多项分布 (Multinomial)
    # consistent, inconsistent, neutral 是观测到的频数向量
    consistent[1:4]   ~ dmulti(predprob[1,1:4], Nsubj[1]) 
    inconsistent[1:4] ~ dmulti(predprob[2,1:4], Nsubj[2]) 
    neutral[1:4]      ~ dmulti(predprob[3,1:4], Nsubj[3]) 

    # 预测概率计算 (对应书中的 Table 8.2)
    # 一致条件 (Consistent condition)
    predprob[1,1] <- (1 + p + q - p*q + 4 * p*c)/6 # Row 1
    # ... 其他行的计算公式 ...
}
```

在 R 中 (`codeFromBook/Chapter8/wagenaar.R`)，我们可以定义辅助函数来分析后验分布：

```r
# 计算参数大于 0.5 的概率
mean(allpost(mcmcfin,"c") > .5)
```

## 5. 最佳实践 (In Vivo)

John Kruschke 建议在使用 JAGS 时注意以下几点：

1.  **有效样本量 (Effective Sample Size, ESS)**：由于 MCMC 样本之间存在自相关，实际包含的信息量小于样本总数。应确保 ESS 至少达到 10,000 以获得稳定的 95% HDI (最高密度区间)。
2.  **代码可读性**：虽然 JAGS 允许任意顺序，但建议按照 **数据 -> 似然 -> 参数关系 -> 先验** 的逻辑顺序编写模型文件。
3.  **模型图 (Diagrams)**：绘制清晰的模型图（如 Kruschke 风格的图）有助于理解模型结构和编写代码。

---
**总结**：本章通过 JAGS 展示了贝叶斯参数估计的通用流程。无论是简单的正态分布，还是复杂的认知模型 (SDT, MPT)，核心步骤都是一样的：定义参数的先验，定义数据生成的似然函数，然后让 JAGS 自动进行 MCMC 采样。
