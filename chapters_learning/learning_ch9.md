# 第 9 章：多层或分层建模 (Multilevel or Hierarchical Modeling)

在第 5 章中，我们讨论了如何处理多个参与者的数据：要么分别拟合每个人的数据（可能噪声大），要么拟合平均数据（可能产生聚合偏差）。本章介绍第三种，也是通常最优的方法：**分层建模 (Hierarchical Modeling)**。

## 1. 理论背景 (Theory)

### 1.1 核心概念

分层模型的核心思想是：虽然每个参与者都有自己独特的参数（如 $d'_i$），但这些参数并不是完全独立的，而是来自同一个**父分布 (Parent Distribution)**（或称超先验分布）。

*   **个体层 (Individual Level)**：描述每个参与者的行为（例如，参与者 $i$ 的辨别力 $d'_i$）。
*   **群体层 (Group Level)**：描述个体参数的分布规律（例如，所有人的 $d'$ 服从均值为 $\mu_d$，标准差为 $\sigma_d$ 的正态分布）。

### 1.2 收缩效应 (Shrinkage)

分层建模的一个重要特性是**收缩效应**。由于假设所有人的参数来自同一个父分布，模型会将那些极端或不可靠的个体估计值向群体均值“拉拢”。这被称为 Stein's Paradox，但在统计上，这种做法通常能提供更准确的预测，因为它利用了群体信息来约束个体估计。

## 2. 分层信号检测论模型 (Hierarchical Signal Detection Theory)

### 2.1 模型形式化

我们将第 8 章的 SDT 模型扩展为分层模型。
对于第 $i$ 个参与者：
$$d'_i \sim N(\mu_d, \tau_d)$$
$$b_i \sim N(\mu_b, \tau_b)$$

其中 $\mu_d, \mu_b, \tau_d, \tau_b$ 是父分布的参数（超参数）。

### 2.2 代码实现

**JAGS 模型** (`codeFromBook/Chapter9/SDThierarch.j`)：

```plaintext
model{
	# 父分布的先验 (Priors for parent distributions)
	mud ~ dnorm(0, epsilon) 
	mub ~ dnorm(0, epsilon)
	taud ~ dgamma(epsilon, epsilon)
	taub ~ dgamma(epsilon, epsilon)  
	
	# 对 n 个参与者进行建模
	for (i in 1:n) { 
		# 个体参数从父分布中采样
		d[i] ~ dnorm(mud, taud) 
		b[i] ~ dnorm(mub, taub)
	
		# 预测命中率和虚报率
		phih[i] <- phi( d[i]/2 - b[i])   
		phif[i] <- phi(-d[i]/2 - b[i])
	
		# 观测数据
		h[i] ~ dbin(phih[i], sigtrials) 
		f[i] ~ dbin(phif[i], noistrials)
	}
}
```

**R 脚本** (`codeFromBook/Chapter9/SDThierarch.R`)：

```r
# ... 数据模拟 ...
# 初始化 JAGS 模型
sdtjh <- jags.model("SDThierarch.j", 
                   data = list("epsilon"=0.001,
                               "h"=h, "f"=f, "n"=n, ...),
                   n.chains=4)  
# ... 采样 ...
```

## 3. 分层遗忘模型 (Hierarchical Modeling of Forgetting)

我们研究记忆随时间衰减的规律，比较指数函数和幂函数。

### 3.1 指数遗忘模型 (Exponential Forgetting)

**形式化**：
$$\theta_{t} = a + (1 - a) \times b \times e^{-\alpha \times t}$$
其中 $a$ 是渐近线，$b$ 是初始记忆强度，$\alpha$ 是衰减率。

**代码实现** (`codeFromBook/Chapter9/hierarchforgexp.j`)：

```plaintext
model{
  # 父分布先验
  mualpha  ~ dunif(0,1)
  # ... 其他超参数 ...
  
  # 个体参数采样
  for (i in 1:ns){
    alpha[i] ~ dnorm(mualpha, taualpha)T(0,1)  
	a[i]     ~ dnorm(mua, taua)T(0,1)
	b[i]     ~ dnorm(mub, taub)T(0,1)  
  }
  
  # 预测每个参与者在每个时间滞后 (lag) 的回忆概率
  for (i in 1:ns){
    for (j in 1:nt){
	    theta[i,j] <- a[i]+(1-a[i])*b[i]*exp(-alpha[i]*t[j]) 
    }
  }
  # ... 似然函数 ...
}
```

### 3.2 幂函数遗忘模型 (Power Forgetting)

**形式化**：
$$\theta_{t} = a + (1 - a) \times b \times (1 + t)^{-\beta}$$

**代码实现** (`codeFromBook/Chapter9/hierarchforgpow.j`)：

只需将预测公式修改为：
```plaintext
theta[i,j] <- a[i]+(1-a[i])*b[i]*pow((t[j]+1), -beta[i]) 
```

## 4. 分层跨期选择模型 (Hierarchical Inter-Temporal Choice)

研究人们如何在“现在的较小奖励”和“未来的较大奖励”之间做选择。

### 4.1 模型形式化

**双曲贴现 (Hyperbolic Discounting)**：
$$V^B = \frac{B}{1 + kD}$$
其中 $V^B$ 是延迟奖励 $B$ 的主观价值，$D$ 是延迟时间，$k$ 是贴现率。

**选择概率**：
$$P(\text{Choose } B) = \Phi\left(\frac{V^B - V^A}{\alpha}\right)$$
其中 $\alpha$ 是决策敏锐度 (acuity)。

### 4.2 代码实现

**JAGS 模型** (`codeFromBook/Chapter9/hierarchicalITC.j`)：

```plaintext
model{
    # 群体层超先验
    groupkmu        ~ dnorm(0, 1/100)
    # ...

    # 参与者层参数
    for (p in 1:nsubj){
      k[p]        ~ dnorm(groupkmu, 1/(groupksigma^2)) T(0,)
      alpha[p]    ~ dnorm(groupALPHAmu, 1/(groupALPHAsigma^2)) T(0,)

      for (t in 1:T) {
        # 计算主观价值
        VA[p,t] <- A[p,t] / (1+k[p]*DA[p,t])  
        VB[p,t] <- B[p,t] / (1+k[p]*DB[p,t])  

        # 心理物理函数计算选择概率
        P[p,t] <-  phi( (VB[p,t]-VA[p,t]) / alpha[p] )  

        # 观测反应 (伯努利分布)
        R[p,t] ~ dbern(P[p,t]) 
      }
    }
}
```

**R 数据处理** (`codeFromBook/Chapter9/hierarchicalITC.R`)：
这个例子展示了如何处理复杂的实验数据结构。代码使用了 `vapply` 和自定义函数 `grabfun` 将长格式的数据框转换为 JAGS 所需的矩阵格式（行=参与者，列=试次）。

## 5. 分层最大似然建模 (Hierarchical Maximum Likelihood Modeling)

除了贝叶斯方法，我们也可以使用频率学派的方法（如广义线性混合模型 GLMM）来实现分层模型。

### 5.1 信号检测论的 GLMM 实现

SDT 模型可以等价于一个 Probit 回归模型：
$$Probit(P(r=1|X)) = -c + d'X$$
其中 $X$ 是信号呈现与否 (0/1)。

### 5.2 代码实现

使用 R 的 `lme4` 包 (`codeFromBook/Chapter9/MLhierarchSDT.R`)：

```r
require(lme4)
# ... 数据准备 ...

# 重新参数化，使得截距代表 -b (偏好)，斜率代表 d' (辨别力)
rmstim <- stim-.5 
reparmstim <- cbind(-1, rmstim)   
colnames(reparmstim) <- c("_b", "_d'")

# 拟合 GLMM 模型
# (1+rmstim|subj) 表示截距和斜率都随受试者 (subj) 随机变化
mlhierarchSDTrp <- glmer(resp ~ reparmstim-1 + (1+rmstim|subj), family=binomial(probit))
summary(mlhierarchSDTrp)
```

这种方法的优点是计算速度快，利用了成熟的统计软件包，但对于非常复杂的非线性模型，贝叶斯方法（JAGS）通常更灵活。
