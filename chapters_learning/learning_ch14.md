# 第 14 章 选择反应时间模型 (Models of Choice Response Time)

在认知心理学中，快速选择任务（Speeded Choice Task）是最常见的实验范式之一。例如，判断屏幕上的点是向左还是向右移动，或者判断一串字母是单词还是非词。这类任务产生两个主要因变量：**反应时间 (Response Time, RT)** 和 **准确率 (Accuracy)**。

本章介绍了一类能够同时解释 RT 和准确率的模型——**序列采样模型 (Sequential Sampling Models)**。这类模型假设决策是一个随时间积累证据的过程，直到证据达到某个阈值。

## 1. 理论背景 (Theory)

### 1.1 速度-准确率权衡 (Speed-Accuracy Tradeoff)
在做决策时，我们可以通过牺牲速度来提高准确率，或者通过牺牲准确率来提高速度。传统的分析方法往往难以同时处理这两个指标。序列采样模型通过**决策边界 (Decision Boundary)** 的概念优雅地解释了这一现象：边界越宽，需要积累的证据越多，反应越慢但越准确；边界越窄，反应越快但越容易出错。

### 1.2 Ratcliff 扩散模型 (Diffusion Model)
这是最著名的序列采样模型。它假设证据积累过程是一个带噪声的维纳过程（Wiener Process）。证据在两个边界（例如“左”和“右”）之间游走，直到触碰其中一个边界。
模型不仅能解释平均 RT，还能解释 RT 的分布形状（通常是右偏的），以及错误反应的 RT。

### 1.3 线性弹道累加器 (Linear Ballistic Accumulator, LBA)
LBA 是另一种流行的模型。与扩散模型不同，LBA 假设证据积累是**弹道式 (Ballistic)** 的，即一旦在一个试次中确定了漂移率，积累过程就是线性的、无噪声的。随机性来自于试次间漂移率和起始点的变异。LBA 在计算上比扩散模型更简单。

## 2. 模型形式化 (Formalization)

### 2.1 扩散模型参数
Ratcliff 扩散模型通常由 7 个参数决定：
1.  **漂移率 (Drift Rate, $\nu$)**：证据积累的平均速度，反映了从刺激中提取信息的效率（即任务难度或个体能力）。
2.  **边界分离 (Boundary Separation, $a$)**：两个决策边界之间的距离，反映了决策的谨慎程度（速度-准确率权衡）。
3.  **起始点 (Starting Point, $z$)**：证据积累的初始位置，反映了对某个选项的偏好。
4.  **非决策时间 (Non-decision Time, $T_{er}$)**：编码刺激和执行运动反应所需的时间。
5.  **跨试次变异参数**：
    *   $\eta$：漂移率的标准差。
    *   $s_z$：起始点的范围。
    *   $s_t$：非决策时间的范围。

### 2.2 LBA 模型参数
LBA 模型假设每个选项都有一个独立的累加器。
1.  **漂移率 (Drift Rate, $d$)**：服从正态分布 $N(v, s)$。
2.  **起始点 (Start Point)**：服从均匀分布 $U[0, A]$。
3.  **阈值 (Threshold, $b$)**：当任一累加器达到 $b$ 时做出反应。
4.  **反应时间**：$RT = \frac{b - \text{start}}{rate} + t_0$。

## 3. 代码实现 (Implementation)

本章主要使用 R 语言的 `rtdists` 包来进行模型的模拟和拟合。

### 3.1 可视化分位数概率函数 (Quantile Probability Functions, QPF)

QPF 是一种同时展示 RT 分布和准确率的有效方法。横坐标表示反应比例（左侧为错误反应，右侧为正确反应），纵坐标表示 RT 的分位数（如 0.1, 0.3, 0.5, 0.7, 0.9）。

代码 `diffusionModelpredQPF.r` 展示了如何绘制 QPF。

```r
library(rtdists)

# 定义绘制 QPF 的函数
qpf <- function(a,v,t0,sz,sv,st0) {
  d  <- 0       # 无偏好
  z  <- 0.5*a   # 起始点在中间
  
  # 计算达到上边界（正确）和下边界（错误）的极限概率 (RT = Inf)
  maxpUp <- pdiffusion(rep(Inf, length(v)), response="upper", 
                       a=a,v=v,t0=t0,z=z,d=d,sz=sz,sv=sv,st0=st0,s=0.1,precision=1)      
  maxpLr <- pdiffusion(rep(Inf, length(v)), response="lower", 
                       a=a,v=v,t0=t0,z=z,d=d,sz=sz,sv=sv,st0=st0,s=0.1,precision=1)      
  
  # 计算 RT 分位数
  qtiles <- seq(from=.1, to=.9, by=.2)
  # ... (计算分位数并绘图的代码，使用 qdiffusion 函数) ...
}

# 设置不同的漂移率 (v) 代表不同的任务难度
v  <- c(.042,.079,.133,.227,.291,.369)

# 绘制不同参数组合下的 QPF
qpf(a=.11, v, t0=0.3, sz=0,    sv=0.0,  st0=0.2) # 基准
qpf(a=.16, v, t0=0.3, sz=0.07, sv=0.12, st0=0.2) # 增加边界距离 (更谨慎)
```

### 3.2 拟合扩散模型

拟合扩散模型通常涉及最大化似然函数。由于扩散模型的似然函数计算较复杂，`rtdists` 包提供了 `ddiffusion` 函数来计算密度。

代码 `diffusionModelFit.r` 展示了基本的拟合过程。

```r
# 定义负对数似然函数
diffusionloglik <- function(pars, rt, response) 
{
  # 使用 tryCatch 处理可能的数值错误
  likelihoods <- tryCatch(ddiffusion(rt, response=response,                 
                          a=pars["a"], 
                          v=pars["v"], 
                          t0=pars["t0"], 
                          z=0.5*pars["a"], # 假设起始点在中间
                          sz=pars["sz"], 
                          st0=pars["st0"], 
                          sv=pars["sv"],s=.1,precision=1),
                        error = function(e) 0)      
  if (any(likelihoods==0)) return(1e6) # 惩罚无效参数
  return(-sum(log(likelihoods)))
}  

# 1. 生成模拟数据
genparms <- c(.1,.2,.5,.05,.2,.05)           
names(genparms) <- c("a", "v", "t0", "sz", "st0", "sv") 
rts <- rdiffusion(500, a=genparms["a"], v=genparms["v"], ...)

# 2. 生成参数初始值
sparms <- c(runif(1, 0.01, 0.4), ...)
names(sparms) <- c("a", "v", "t0", "sz", "st0", "sv") 

# 3. 使用 optim 进行优化
fit2rts <- optim(sparms, diffusionloglik, gr = NULL, 
               rt=rts$rt, response=rts$response)
```

### 3.3 拟合 LBA 模型

LBA 模型的拟合过程与扩散模型非常相似，只是使用的函数变为 `dLBA` (计算密度) 和 `rLBA` (生成数据)。

代码 `LBAFitLong.r` 展示了如何拟合具有多个条件（不同漂移率）的 LBA 模型。

```r
# 定义 LBA 的负对数似然函数
LBAloglik <- function(pars, rt, response) 
{
  if (any(pars<0)) return(1e6+1e3*rnorm(1)) # 参数必须为正
  ptrs <- grep("v[1-9]",names(pars)) # 查找所有漂移率参数
  
  likelihoods <- NULL
  # 遍历每个条件
  for (i in c(1:length(ptrs))) {
    likelihoods <- c(likelihoods,
                     tryCatch(dLBA(rt[...],  
                                   response=response[...],                 
                                   A=pars["A"], 
                                   b=pars["b"],
                                   t0=pars["t0"], 
                                   # LBA 需要为每个累加器指定漂移率
                                   # 这里假设正确和错误反应的漂移率和为 1
                                   mean_v=c(pars[ptrs[i]],1-pars[ptrs[i]]), 
                                   sd_v=c(pars["sv"],pars["sv"])),
                              error = function(e) 0))
  }
  return(-sum(log(likelihoods)))
} 

# ... (生成数据和优化过程类似) ...
```

## 4. 运行结果与讨论

1.  **QPF 图解**：运行 `diffusionModelpredQPF.r` 会生成一系列 QPF 图。你会看到，随着漂移率 $\nu$ 的增加（任务变容易），数据点向右上方移动（准确率提高，RT 变快）。增加边界分离 $a$ 会导致分位数之间的间距变大（RT 分布变宽），且整体向右移动（准确率提高）。
2.  **参数恢复**：`diffusionModelFit.r` 和 `LBAFitLong.r` 演示了参数恢复（Parameter Recovery）的过程。即先用已知参数生成数据，再用模型拟合这些数据，看能否还原出原始参数。这是验证模型有效性的重要步骤。结果通常显示，只要数据量足够，这些模型都能很好地恢复出生成参数。
3.  **模型比较**：虽然扩散模型和 LBA 在机制上不同（随机游走 vs. 线性弹道），但它们通常能对同一组数据做出相似的预测。LBA 的优势在于数学形式更简单，计算似然函数更快。

通过这些模型，我们可以将观测到的 RT 和准确率分解为心理学意义明确的潜在过程（如信息处理效率、决策谨慎度），从而更深入地理解人类决策机制。
