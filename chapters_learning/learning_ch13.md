# 第 13 章 神经网络模型 (Neural Network Models)

本章介绍了一类模拟大脑基本运作原理的计算模型——神经网络模型（Neural Network Models），也称为联结主义模型（Connectionist Models）。这类模型不追求精确模拟单个神经元的生物物理特性，而是抽象出神经元之间通过连接（权重）传递兴奋或抑制信号的机制，以此来解释学习、记忆和认知表现。

## 1. 理论背景 (Theory)

神经网络模型的核心思想是：知识不是存储在某个特定的位置，而是分布在神经元之间的连接权重（Weights）中。学习的过程就是调整这些权重的过程。

本章主要介绍了两类模型：
1.  **Hebbian 模型 (Hebbian Models)**：基于“共同激发的神经元连接会增强”这一原理（Cells that fire together, wire together）。它适用于单次学习（One-shot learning）和联想记忆（Associative Memory）。
2.  **反向传播模型 (Backpropagation Models)**：引入了误差驱动的学习机制，通过计算输出误差并将其反向传播回网络，来调整多层网络中的权重。它解决了 Hebbian 模型无法解决的非线性问题（如 XOR 问题），并能学习复杂的内部表征。

## 2. 模型形式化 (Formalization)

### 2.1 Hebbian 学习

在一个简单的两层网络（输入层和输出层）中，输入向量为 $\mathbf{c}$ (cue)，输出向量为 $\mathbf{o}$ (output)。

**检索 (Retrieval)**：
输出单元 $i$ 的激活值 $o_i$ 是所有输入单元 $j$ 的加权和：
$$o_i = \sum_j W_{ij} c_j$$
用矩阵形式表示，即：
$$\mathbf{o} = \mathbf{Wc}$$

**学习 (Learning)**：
Hebbian 规则规定，权重 $W_{ij}$ 的变化量与输入 $c_j$ 和输出 $o_i$ 的乘积成正比：
$$\Delta W_{ij} = \alpha c_j o_i$$
其中 $\alpha$ 是学习率。用矩阵形式表示，权重的更新等于目标输出向量 $\mathbf{o}$ 与输入向量 $\mathbf{c}$ 的**外积 (Outer Product)**：
$$\mathbf{W} = \mathbf{W} + \alpha \mathbf{o} \mathbf{c}^T$$

### 2.2 Brain-State-in-a-Box (BSB) 模型

BSB 是一种自联想（Auto-associator）模型，具有反馈连接。其状态更新公式为：
$$\mathbf{u}(t+1) = G[\beta \mathbf{u}(t) + \epsilon \mathbf{W} \mathbf{u}(t)]$$
其中 $G$ 是一个限制函数，将激活值限制在 $[-1, 1]$ 的“盒子”内。这使得模型能够进行模式补全（Pattern Completion）和分类。

### 2.3 反向传播 (Backpropagation)

反向传播模型通常包含输入层、隐藏层和输出层。激活函数通常是非线性的（如 Logistic 函数）：
$$f(a) = \frac{1}{1 + \exp(-a)}$$

**Delta 规则 (Delta Rule)**：
权重的更新旨在最小化输出误差。对于输出层单元，权重更新公式为：
$$\Delta W_{ij} = \eta (t_i - o_i) o_i (1 - o_i) c_j$$
其中 $(t_i - o_i)$ 是误差，$o_i(1-o_i)$ 是 Logistic 函数的导数。对于隐藏层，误差需要从输出层反向传播回来。

## 3. 代码实现 (Implementation)

### 3.1 简单的 Hebbian 联想器

我们首先看一个最简单的 Hebbian 学习实现，代码源自 `simpleHebb.R`。

```r
# 定义输入 (stim) 和 目标输出 (resp)
stim <- list(c(1,-1,1,-1),
             c(1,1,1,1))
resp <- list(c(1,1,-1,-1),
             c(1,-1,-1,1))

n <- 4 # 输入单元数
m <- 4 # 输出单元数
W <- matrix(rep(0,m*n), nrow=m) # 初始化权重矩阵
alpha <- 0.25

# 学习过程：双重循环遍历所有输入输出对
for (pair in 1:2){ 
  for (i in 1:m){ 
    for (j in 1:n){ 
      # Hebbian 规则: W_ij = W_ij + alpha * input_j * output_i
      W[i,j] <- W[i,j] + alpha*stim[[pair]][j]*resp[[pair]][i] 
    }
  }
}

# 矩阵形式的学习（更高效）
# W2 <- resp[[1]]%*%t(stim[[1]]) + resp[[2]] %*% t(stim[[2]])

# 测试阶段：给定第一个刺激，计算输出
o <- rep(0,m)
for (i in 1:m){
  for (j in 1:n){
    o[i] <- o[i] + W[i,j]*stim[[1]][j]
  }
}
# 结果 o 应该接近 resp[[1]]
```

### 3.2 泛化与优雅退化 (Generalization & Graceful Degradation)

神经网络的一个重要特性是**泛化能力**（对相似但不完全相同的输入做出正确反应）和**优雅退化**（在部分权重受损时性能不会突然崩溃）。

代码 `HebbGraceful.R` 演示了泛化能力。它训练网络学习一组向量，然后用不同相似度（`stimSim`）的探针进行测试。

```r
# ... (初始化代码略) ...

# 学习阶段：使用矩阵运算
for (litem in 1:listLength){
  c <- stim1[[litem]]
  o <- resp1[[litem]]
  # 外积更新权重: W = W + alpha * o * c^T
  W <- W + alpha*o%*%t(c)
}

# 测试阶段：使用不同相似度的探针
for (stimSimI in 1:length(stimSimSet)){
  stimSim <- stimSimSet[stimSimI]
  
  # 创建测试刺激：混合原始刺激和随机噪声
  stim2 <- {}
  for (litem in 1:listLength){
    svec <- sign(rnorm(n))
    mask <- runif(n)<stimSim # 控制相似度
    stim2 <- c(stim2,list(mask*stim1[[litem]] + (1-mask)*svec))
  }
  
  # 计算余弦相似度 (Cosine Similarity)
  tAcc <- 0
  for (litem in 1:listLength){
    c <- stim2[[litem]]
    o <- W %*% c # 矩阵乘法计算输出
    tAcc <- tAcc + cosine(as.vector(o),resp1[[litem]])
  }
  # ...
}
```

### 3.3 Brain-State-in-a-Box (BSB)

代码 `BSBprobmatch.R` 模拟了 BSB 模型在分类任务中的表现。模型学习了两个正交向量 A 和 B，然后测试它如何对混合了 A 和 B 的噪声输入进行分类。

```r
# ... (初始化) ...

# 学习两个正交向量
for (i in 1:2){
  W <- W + alpha*v[[i]]%*%t(v[[i]])
}

# BSB 动力学更新循环
for (t in 1:maxUpdates){
  ut <- u
  # 状态更新: u(t+1) = beta * u(t) + epsilon * W * u(t)
  u <- beta*u + epsilon * W%*%u 
  
  # 限制函数 (Squashing function): 限制在 [-1, 1]
  u[u > 1] <- 1
  u[u < -1] <- -1
  
  # 检查收敛
  if (all(abs(u-ut)<tol)){
    break
  }
}
```

### 3.4 反向传播与规则/不规则映射

代码 `BPregularity.R` 演示了如何用反向传播网络学习英语发音中的规则（Regular）和不规则（Irregular）映射。这是一个经典的应用，展示了单一机制如何处理看似需要双重机制（规则 vs. 记忆）的任务。

```r
# Logistic 激活函数
logistic_act <- function(x){
  return(1/(1+exp(-x)))
}

# ... (数据生成：大部分数据遵循规则，少量例外) ...

for (sweep in 1:nTrain){
  # 前向传播 (Forward Pass)
  net <- Wih %*% cue
  act_hid <- logistic_act(net + Bh) # 隐藏层激活
  
  net <- Who %*% act_hid
  act_out <- logistic_act(net + Bo) # 输出层激活
  
  # 计算输出层误差项 (Delta)
  # d_out = (target - output) * f'(net)
  d_out <- (target-act_out)*act_out*(1-act_out) 
  
  # 计算隐藏层-输出层权重的梯度
  dWho <- eta * d_out%*%t(act_hid) + m*dWho 
  
  # 反向传播误差到隐藏层
  d_hid <- t(Who)%*%d_out * act_hid*(1-act_hid) 
  
  # 计算输入层-隐藏层权重的梯度
  dWih <- eta * d_hid%*%t(cue) + m*dWih
  
  # 更新权重
  Who <- Who + dWho 
  Wih <- Wih + dWih
  # ...
}
```

## 4. 运行结果与讨论

1.  **Hebbian 泛化**：运行 `HebbGraceful.R` 会生成一个图表，显示随着测试探针与原始刺激相似度（Stimulus-Cue Similarity）的降低，模型输出与正确答案的余弦相似度也逐渐降低。这表明模型具有良好的泛化能力，即使输入有噪声也能恢复出大致正确的输出。
2.  **BSB 分类**：`BSBprobmatch.R` 的结果显示，BSB 模型能够根据输入向量更接近 A 还是 B，将其“吸入”对应的吸引子状态（Attractor State）。即使输入含有大量噪声，模型也能通过迭代清除噪声，实现分类。
3.  **规则 vs. 不规则学习**：`BPregularity.R` 模拟结果通常显示，规则项（Regular items）的学习速度很快，误差迅速下降；而不规则项（Irregular items）的学习速度较慢，且最终误差通常略高于规则项。这与人类学习语言（如动词过去式）的行为模式相符，表明神经网络可以通过调整权重来同时编码一般规则和特例，而不需要两个独立的系统。

神经网络模型通过简单的数学原理（矩阵运算、微积分），展现了强大的学习和适应能力，为理解人类认知提供了一个自下而上的视角。
