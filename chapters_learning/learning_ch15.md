# Chapter 15: Models in Neuroscience (Reinforcement Learning)

本章探讨了计算模型如何连接认知心理学与神经科学，特别是**强化学习 (Reinforcement Learning, RL)** 模型如何成功地解释了大脑中的多巴胺 (Dopamine) 活动。我们将重点关注两个核心概念：用于解决多臂老虎机问题的 Rescorla-Wagner 规则，以及用于解释多巴胺相位放电的**时间差分学习 (Temporal Difference Learning, TD Learning)**。

## 1. 理论背景 (Theory)

### 认知神经科学中的建模
计算模型在神经科学中扮演着桥梁的角色，连接了 Marr 提出的不同分析层次：
- **计算层 (Computational)**: 系统试图解决什么问题？（例如：最大化长期奖励）
- **算法层 (Algorithmic)**: 使用什么规则来解决问题？（例如：Rescorla-Wagner 规则，TD 学习）
- **实现层 (Implementational)**: 物理硬件如何执行这些算法？（例如：基底神经节和多巴胺神经元的放电模式）

### 强化学习与多巴胺
强化学习模型假设个体通过试错来学习行为的价值。
1.  **Rescorla-Wagner 模型**: 这是一个经典的联结主义学习规则，认为学习是由**预测误差 (Prediction Error)** 驱动的。即：实际获得的奖励与预期奖励之间的差值。
2.  **多巴胺的预测误差假说**: 神经生理学实验（如 Schultz et al., 1997）发现，中脑多巴胺神经元的放电模式与 RL 模型中的预测误差高度一致：
    - 在学习初期，奖励出现时多巴胺神经元放电（正误差）。
    - 随着学习进行，奖励被完全预测，奖励出现时不再有额外的放电（误差为 0）。
    - 此时，预测奖励的**线索 (CS)** 出现时，多巴胺神经元开始放电（预测本身变成了奖励信号）。
    - 如果预期奖励未出现，多巴胺神经元放电会受到抑制（负误差）。

## 2. 模型形式化 (Formalization)

### 2.1 Rescorla-Wagner 规则 (Delta Rule)
在简单的老虎机任务 (Bandit Task) 中，我们通过更新动作价值 $Q$ 来学习：

$$ Q_{t+1}(a) = Q_t(a) + \alpha \cdot \delta_t $$

其中：
- $Q_t(a)$ 是在时间 $t$ 对动作 $a$ 的价值估计。
- $\alpha$ 是学习率 (Learning Rate)，控制更新步长。
- $\delta_t$ 是**预测误差 (Prediction Error)**：

$$ \delta_t = r_t - Q_t(a) $$

即：(实际奖励) - (预期奖励)。

### 2.2 时间差分学习 (Temporal Difference Learning)
Rescorla-Wagner 规则通常只在试次结束时更新。而 TD 学习允许在每一个时间步 (Time Step) 进行更新，这对于解释多巴胺的精细时间动态至关重要。

TD 误差 $\delta_t$ 定义为：

$$ \delta_t = r_t + \gamma V(S_{t+1}) - V(S_t) $$

其中：
- $r_t$ 是当前时刻获得的奖励。
- $V(S_t)$ 是当前状态的价值估计。
- $V(S_{t+1})$ 是下一时刻状态的价值估计。
- $\gamma$ 是折扣因子 (Discount Factor)，衡量未来奖励的权重。

这个公式的含义是：**当前的预测误差 = 当前奖励 + 对未来的折现预测 - 当前的预测**。

## 3. 代码实现 (Implementation)

### 3.1 多臂老虎机任务 (`banditExample.R`)

该代码模拟了一个 2 臂老虎机任务，代理 (Agent) 必须在两个选项中选择以最大化奖励。

首先，初始化环境和参数：
```r
# 引用自: computational-modelling-master/codeFromBook/Chapter15/banditExample.R

nTrials <- 1000
# 生成两个选项的奖励分布 (正态分布)
r1 <- rnorm(nTrials, mean = 5, sd = 1)
r2 <- rnorm(nTrials, mean = 5.5, sd = 1) # 选项 2 平均奖励更高
r <- rbind(r1, r2)

epsilon <- 0.1 # 探索率 (Epsilon-greedy)
alpha = 0.1    # 学习率
```

接下来是核心的学习循环。这里使用了 **Epsilon-Greedy** 策略进行决策，并使用 Delta 规则更新 Q 值：

```r
  for (i in 1:nTrials){
    
    # 1. 决策: Epsilon-greedy 策略
    if (runif(1) < epsilon){
      # 探索 (Explore): 随机选择
      a <- sample(2, 1)
    } else {
      # 利用 (Exploit): 选择当前 Q 值最高的动作
      a <- which.max(Q)[1]
    }
    
    # 2. 学习: 更新 Q 值
    # Q[a] <- Q[a] + alpha * (r[a,i] - Q[a])
    # 这里的 (r[a,i] - Q[a]) 就是预测误差 delta
    Q[a] <- Q[a] + alpha*(r[a,i] - Q[a])
    
    QthisRun[,i] <- Q
  }
```
这段代码展示了最基础的强化学习过程：根据当前的 Q 值做出选择，观察奖励，然后修正 Q 值。

### 3.2 多巴胺的相位模拟 (`TDphasic.R`)

这段代码模拟了经典的巴甫洛夫条件反射实验中的多巴胺反应。在一个试次 (Trial) 中包含多个时间步 (Time Steps)。刺激 (CS) 在第 5 步出现，奖励 (Reward) 在最后一步出现。

```r
# 引用自: computational-modelling-master/codeFromBook/Chapter15/TDphasic.R

nTrials <- 40  # 试次数量
nSteps <- 25   # 每个试次的时间步数
stimStep <- 5  # 刺激出现的时间步
# 奖励在 s == nSteps 时出现

gamma <- 1     # 折扣因子
alpha <- 0.5   # 学习率
```

核心循环计算每个时间步的 TD 误差。注意代码中如何通过 `x` (当前状态) 和 `x1` (下一状态) 来计算价值 `Vt` 和 `Vt1`：

```r
  for (s in 1:nSteps){
    
    # ... (省略了设置 x 和 x1 向量的代码，它们用于指示当前时刻) ...
    
    # 设定奖励 r: 只有在最后一步才有奖励
    if (s==nSteps){
      r=1
    } else {
      r=0
    }
    
    # 计算 t 时刻和 t+1 时刻的预测价值
    # w 是权重向量，x 是状态向量 (one-hot encoding)
    Vt <- sum(w*x)
    Vt1 <- sum(w*x1)
    
    # 计算 TD 预测误差 (Prediction Error)
    # dt = r + gamma * V(t+1) - V(t)
    dt <- r + gamma*Vt1 - Vt
    
    # 记录误差用于绘图
    alld[trial,s] <- dt
    
    # 累积权重更新量 (在试次结束时统一更新，这是 batch update 的一种形式)
    sumd <- sumd + x*dt
  }
  # 更新权重
  w <- w + alpha * sumd
```

## 4. 运行结果与讨论

### 结果分析
运行 `TDphasic.R` 会生成一个展示预测误差随时间变化的图表。
- **Trial 1 (早期)**: 预测误差 $\delta_t$ 仅在奖励出现时刻 (Time step 25) 为正值。因为此时 $V(S_t)$ 接近 0，奖励是完全意外的。
- **Trial 12 (中期)**: 预测误差开始向前移动。因为 $V(S_{reward})$ 增加，导致 $V(S_{reward-1})$ 通过 $r + \gamma V(S_{reward})$ 的更新也开始增加。误差逐渐从奖励时刻向刺激时刻“反向传播”。
- **Trial 40 (晚期)**: 预测误差主要出现在刺激呈现时刻 (Time step 5)。此时，刺激本身已经成为了奖励的可靠预测器。而在奖励实际出现的时刻，由于预测已经非常准确 ($V(S_t) \approx r$)，误差 $\delta_t$ 接近于 0。

### 神经科学意义
这个模拟完美复现了 Schultz 等人观察到的多巴胺神经元放电模式：
1.  **未预期奖励**: 引起多巴胺爆发 (Trial 1)。
2.  **已学习奖励**: 奖励本身不引起爆发，但预测线索 (CS) 引起爆发 (Trial 40)。
3.  **误差反向传播**: 解释了价值信号是如何在时间上转移的。

这证明了 TD 学习算法不仅是一个高效的计算机算法，也是理解大脑奖赏系统运作机制的有力工具。
