# petitorch

petitorch 是一个基于纯 Python 和 NumPy 实现的微型深度学习框架。

本项目的核心目标并非追求极致的运算性能，而是通过最小化的代码量，严格复刻 PyTorch 底层的核心机制，包括动态计算图（Define-by-Run）、分发器机制（Dispatcher）以及基于伴随方法（Adjoint Method）的自动求导引擎。

技术路线 (Architecture & Design)

petitorch 在架构设计上严格遵循“高内聚、低耦合”的设计哲学，将数据载体、计算图拓扑与数学运算彻底分离。

1. 核心解耦设计

Tensor (数据载体)：仅作为数据（NumPy Array）和梯度状态的容器。Tensor 本身不维护计算图的上下游拓扑关系，仅通过重载魔法方法（如 __add__）充当调用底层算子的入口。

Function (算子基类与分发器)：所有数学操作均继承自该基类。其 apply 方法充当**动态分发器（Dispatcher）**的角色。它负责在前向传播时检查输入张量的 requires_grad 状态，动态决定是否介入 Autograd 引擎。

Context (闭包与图节点)：在建图模式下，apply 会实例化 Context。它负责保存前向传播的张量（用于 VJP 计算）以及记录图的连接边（prev_edges）。

2. 零开销推理 (Zero-Overhead Inference)

框架原生支持推理模式优化。当所有输入的 requires_grad 均为 False 时，分发器会跳过计算图的构建，并注入 NoOpContext（空对象模式）。此时算子仅执行 NumPy 原生计算，不产生任何由于保存前向数据带来的内存泄漏或计算图开销。

3. 反向传播引擎 (Autograd Engine)

拓扑排序：基于深度优先搜索（DFS）对有向无环图（DAG）进行后序遍历，确保梯度计算顺序严格遵循依赖关系。
向量-雅可比乘积 (VJP)：引擎不进行标量链式求导，而是通过传递全尺寸的梯度张量（error），调用各算子定义的伴随算子（backward 方法）直接计算雅可比矩阵的左乘。

多分支梯度累加：支持同一个 Tensor 作为多个算子输入的情况，引擎会在反向传播时自动对图节点的梯度进行正确累加。

当前完成状态 (Status)

目前已完成底层 Autograd 引擎与基础张量操作，能够正确运行带有分支拓扑的动态计算图。

基础核心模块

Tensor: 核心数据结构与魔法方法拦截。

Function: 动态分发器、图节点构建逻辑。

Context / NoOpContext: 状态闭包与推理期内存优化。

自动求导引擎

标量反向传播入口 (loss.backward())。

基于 DFS 的计算图拓扑排序。

节点间梯度分发与叶子节点梯度累加机制。

基础算子库 (Ops)

加法 (Add)：包含广播机制的逆运算（Unbroadcast）处理。

乘法 (Mul)：Element-wise VJP 实现。

求和 (Sum)：用于将高维张量降维至标量，触发引擎引擎入口。

神经网络高层 API (待实现)

nn.Module 与参数管理 (nn.Parameter)。

基础层实现（如 nn.Linear）与激活函数（ReLU 等）。

损失函数（MSE, CrossEntropy）。

优化器 (待实现)

optim.SGD 基础梯度下降。

基本使用示例

petitorch 的 API 设计与 PyTorch 保持高度一致。


```
import numpy as np
from petitorch.tensor import Tensor

# 1. 定义带有梯度的叶子节点
w = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)
x = Tensor(np.array([[1.0], [2.0]]), requires_grad=False)

# 2. 前向传播 (动态构建计算图)
# 执行矩阵相乘并求和
logits = w @ x 
loss = logits.sum()

# 3. 反向传播
loss.backward()

# 4. 查看梯度
print("w.grad:", w.grad)
```