import sys
import os
# 强行把当前文件所在目录的上一级，加入到 Python 的环境变量中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from petitorch.tensor import Tensor

def test_engine():
    print("=========================================")
    print("🚀 开始测试 PetiTorch 核心引擎 🚀")
    print("=========================================")

    # 1. 初始化变量 (测试需要梯度的叶子节点)
    # 我们用大小为 1 的数组来模拟标量，这样可以通过 engine 的防呆测试
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    # 偷偷混入一个不需要梯度的常量，测试引擎会不会崩溃
    c_const = Tensor([4.0], requires_grad=False)

    print(f"初始输入: a={a.data}, b={b.data}, c_const={c_const.data}")

    # 2. 构建一张刁钻的计算图 (Forward Pass)
    # 数学公式: loss = (a * b) + (a * a) + c_const
    # 为什么刁钻？因为变量 'a' 在图中分叉了三次！
    # 如果拓扑排序或梯度分发写错了，a 的梯度绝对算不对！
    
    term1 = a * b
    term2 = a * a
    loss = term1 + term2 + c_const

    print(f"\n✅ 前向传播结果 loss: {loss.data} (期望值: [14.])")
    
    # 3. 引爆反向传播！(Backward Pass)
    print("\n💥 执行 loss.backward()...")
    loss.backward()

    # 4. 见证奇迹的时刻：数学真实验证
    # 数学求导解析解：
    # loss = a*b + a^2 + c
    # d(loss)/da = b + 2a = 3.0 + 2*(2.0) = 7.0
    # d(loss)/db = a = 2.0
    # d(loss)/dc = 0 (或者 None，因为它不需要梯度)
    
    print("\n=========================================")
    print("📊 梯度核对 (Gradient Check) 📊")
    print("=========================================")
    print(f"a 的梯度: {a.grad} \t| 期望值: [7.]")
    print(f"b 的梯度: {b.grad} \t| 期望值: [2.]")
    print(f"c_const 的梯度: {c_const.grad} \t| 期望值: None")
    
    if np.allclose(a.grad, [7.0]) and np.allclose(b.grad, [2.0]):
        print("\n🎉 恭喜你！！！拓扑引擎运转完美，梯度计算 100% 正确！🎉")
    else:
        print("\n❌ 梯度计算有误，请回头检查图遍历逻辑或算子求导公式！")

if __name__ == "__main__":
    test_engine()