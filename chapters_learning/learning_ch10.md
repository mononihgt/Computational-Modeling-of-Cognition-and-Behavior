# 第 10 章：模型比较 (Model Comparison)

在前面的章节中，我们学习了如何构建模型以及如何使用最大似然估计（MLE）或贝叶斯方法来估计参数。然而，科学研究的核心不仅仅是拟合一个模型，更在于**比较**不同的理论解释。

本章我们将探讨如何科学地比较模型。一个拟合得更好的模型就一定更好吗？不一定。我们需要在**拟合优度 (Goodness of Fit)** 和 **模型复杂性 (Model Complexity)** 之间找到平衡。

## 1. 理论背景 (Theory)

### 1.1 过度拟合与“非常糟糕的良好拟合”
心理学家 Nihm (1976) 曾发表过一篇讽刺文章，描述了一个“非常糟糕的良好拟合” (Very Bad Good Fit)。他指出，如果你有足够多的参数，你可以拟合任何数据。

这就引出了**过度拟合 (Over-fitting)** 的概念：一个过于复杂的模型不仅捕捉到了数据中的潜在规律（信号），还捕捉到了随机噪声。这样的模型在当前数据集上表现完美，但在预测新数据时会一塌糊涂。

### 1.2 偏差-方差权衡 (Bias-Variance Trade-off)
模型误差可以分解为两部分：
1.  **偏差 (Bias)**：模型太简单，无法捕捉数据的真实形态（欠拟合）。
2.  **方差 (Variance)**：模型太复杂，对数据中的微小波动（噪声）过于敏感。

我们的目标是找到一个“刚刚好”的模型，既能捕捉规律，又忽略噪声。

### 1.3 案例研究：累积前景理论 (Cumulative Prospect Theory, CPT)
为了演示模型比较，本章使用了 Rieskamp (2008) 的风险决策数据。参与者需要在两个赌局（Gamble A vs Gamble B）中进行选择。

我们将比较两个模型：
1.  **累积前景理论 (CPT)**：一个复杂的参数化模型，假设人们根据主观价值和主观概率来做决定。
2.  **优先启发式 (Priority Heuristic)**：一个简单的非参数化模型（或少参数），假设人们通过一系列简单的规则（如“先比较最小收益”）来做决定。

## 2. 模型形式化 (Formalization)

### 2.1 累积前景理论 (CPT)
CPT 是对期望效用理论的修正。它包含两个核心函数：

**价值函数 (Value Function)** $v(x)$：
$$
v(x) = 
\begin{cases} 
x^\alpha & \text{if } x \ge 0 \\
-\lambda(-x)^\beta & \text{if } x < 0 
\end{cases}
$$
*   $\alpha, \beta$：控制价值函数的曲率（通常 $<1$，表示敏感度递减）。
*   $\lambda$：**损失厌恶 (Loss Aversion)** 参数。如果 $\lambda > 1$，表示损失带来的痛苦大于同等收益带来的快乐。

**概率权重函数 (Probability Weighting Function)** $w(p)$：
$$
w(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}
$$
*   $\gamma$ (或 $\delta$ 对于损失)：控制概率权重的曲率。通常呈现倒 S 型，即高估小概率，低估大概率。

**选择规则**：
使用 Softmax (或 Logit) 规则将主观价值差异转化为选择概率：
$$
P(\text{Choose A}) = \frac{e^{\phi \cdot V(A)}}{e^{\phi \cdot V(A)} + e^{\phi \cdot V(B)}}
$$
其中 $\phi$ 是敏感度参数。

### 2.2 模型比较指标

#### 似然比检验 (Likelihood Ratio Test, LRT)
适用于**嵌套模型 (Nested Models)**。如果模型 A 是模型 B 的一个特例（例如，固定模型 B 的某个参数为 1 就得到模型 A），则它们是嵌套的。
统计量 $G^2$ 服从 $\chi^2$ 分布：
$$
G^2 = 2(\ln L_{\text{general}} - \ln L_{\text{restricted}})
$$
自由度 $df$ 等于参数数量之差。

#### 赤池信息准则 (Akaike Information Criterion, AIC)
适用于**非嵌套模型**。AIC 旨在估计模型与“真实”数据生成过程之间的 Kullback-Leibler 距离。
$$
AIC = -2 \ln L + 2K
$$
*   $-2 \ln L$：Deviance（偏差），衡量拟合优度（越小越好）。
*   $2K$：惩罚项，其中 $K$ 是参数数量。

**AIC 越小越好**。它在拟合好坏与模型复杂性之间做了权衡。

## 3. 代码实现 (Implementation)

我们将重点关注 CPT 的实现及其与优先启发式的比较。

### 3.1 CPT 的 R 语言实现
文件 `codeFromBook/Chapter10/cumulPT.R` 定义了 CPT 的核心逻辑。

首先是概率权重函数：
```r
# probability weighting function
probw <- function(p,c){
  return(p^c/
           ((p^c + (1-p)^c)^(1/c))
  )
}
```

然后是计算前景（Prospect）主观价值的函数。注意它分别处理收益（`x >= 0`）和损失（`x < 0`），并应用了累积概率的逻辑（即对排序后的结果进行加权）：

```r
cumulPTv <- function(x, p, alpha=1,beta=1,lambda=1,gamma=1,delta=1){
  # ... (排序代码省略) ...
  
  # ---- deal with x>= 0 (Gains)
  ii <- which(x>=0)
  if (length(ii)>0){
    tx <- x[ii]
    tp <- p[ii]
    vp <- tx^alpha  # 价值函数 (收益)
    
    # 累积概率权重计算 (Rank-dependent utility)
    pp <- {}
    pp[length(tp)] <- probw(tp[length(tp)],gamma)
    if (length(ii)>1){
      for (j in 1:(length(ii)-1)){
        pp[j] <- probw(sum(tp[j:length(tp)]),gamma)-
          probw(sum(tp[(j+1):length(tp)]),gamma)
      }
    }
    sv_pos <- sum(vp*pp)
  } 
  # ... (损失部分类似，使用 lambda 和 beta) ...
}
```

### 3.2 拟合 CPT 并进行似然比检验
文件 `codeFromBook/Chapter10/fitCPT.R` 展示了如何拟合 CPT。

这里进行了两次拟合：
1.  **完整模型**：$\lambda$ 自由估计。
2.  **受限模型**：$\lambda$ 固定为 1（假设没有损失厌恶）。

```r
# 完整模型拟合 (部分代码)
# fit individuals with lambda free
# 参数顺序: alpha, lambda, gamma, delta, phi
tfit <- nmkb(par=startPoints[sp,],
             fn = function(theta) fitCPT(c(theta[1],theta[1],theta[2:5]), ...),
             lower=c(0,0,0,0,0),
             upper=c(1,10,1,1,10), ...)

# 受限模型拟合 (部分代码)
# fit individuals with lambda fixed at 1
# 注意 fitCPT 调用中的第3个参数被硬编码为 1
tfit <- nmkb(par=startPoints[sp,],
             fn = function(theta) fitCPT(c(theta[1],theta[1],1,theta[2:4]), ...),
             lower=c(0,0,0,0),
             upper=c(1,1,1,10), ...)
```

通过比较这两个模型的 Deviance ($-2 \ln L$)，我们可以计算 $\chi^2$ 统计量，判断 $\lambda$ 是否显著异于 1。书中结果显示，虽然总体上有显著差异，但主要是由少数极端被试驱动的。

### 3.3 优先启发式 (Priority Heuristic)
文件 `codeFromBook/Chapter10/priorityHeuristic.R` 实现了这个非参数模型。它不进行复杂的加权求和，而是按顺序比较：

1.  比较最小收益 (Minimum Gain)。如果差异够大，直接选择。
2.  否则，比较最小收益的概率。
3.  否则，比较最大收益。

```r
priorityHeuristic <- function(prospect,alpha=0.1){
    # ...
    # first compare minimums
    max_x <- max(allx)
    mins <- c(min(prospect[[1]]$x),min(prospect[[2]]$x))
    # 如果差异大于最大值的 10%
    if ((abs(mins[1]-mins[2])/max_x)>0.1){
      choice_p <- rep(alpha,2)
      choice_p[which.max(mins)] <- 1-alpha # 选择最小收益较高的那个
      return(choice_p)
    }
    # ... (后续步骤)
}
```
注意：为了能进行最大似然估计，这里引入了一个“颤抖参数” (tremble parameter) $\alpha$，假设人们有 $\alpha$ 的概率随机选择非偏好选项。

## 4. 运行结果与讨论

### 4.1 AIC 比较结果
书中计算了 CPT 和 优先启发式 (PH) 的 AIC 值：

*   **CPT (完整版)**:
    *   Deviance: 5378.41
    *   参数数量 $K$: 5 (每个被试)
    *   $AIC_{CPT} = 5378.41 + 2 \times 5 \times 30 = 5678.41$
*   **优先启发式 (PH)**:
    *   Deviance: 7242.10
    *   参数数量 $K$: 1 (每个被试，即噪音参数 $\alpha$)
    *   $AIC_{PH} = 7242.10 + 2 \times 1 \times 30 = 7302.10$

**结论**：$\Delta AIC = 7302.10 - 5678.41 = 1623.69$。
CPT 的 AIC 远小于 PH。根据 AIC 准则（$\Delta AIC > 10$ 即为强有力证据），在这个数据集上，**CPT 提供了比优先启发式更好的解释**，即使考虑到 CPT 更加复杂（参数更多）。

### 4.2 关键启示
1.  **拟合优度不是唯一标准**：必须考虑模型复杂性。
2.  **AIC 的作用**：它提供了一个通用的标尺，让我们能比较机制完全不同的模型（如代数模型 CPT vs 算法模型 PH）。
3.  **参数识别性 (Identifiability)**：本章末尾还讨论了如果模型参数无法被数据唯一确定（如 Sternberg 记忆扫描模型），模型就是不可识别的。这是建模时需要极力避免的陷阱。
