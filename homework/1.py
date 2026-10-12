import torch

def sec(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

sec("练习一")
x = torch.randn(16, 3, 32, 32)
print("这个shape为（16，3，32，32）的张量每个通道的均值为：", x.mean(dim=(0, 2, 3)))

sec("练习二")
A = torch.randn(5, 4)
B = torch.randn(4, 2)
C = A @ B
print("A@B的形状为", C.shape)

sec("练习三")
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = (x**2).sum()
y.backward()
print("x的梯度为", x.grad)

sec("练习四")
w = torch.randn((3,), requires_grad=True)   # 需要求梯度的东西，加 requires_grad=True
b = torch.tensor((1,), requires_grad=True)
x = torch.randn(4,3)

y = w * x + b          # 前向
print("w.shape =", w.shape)   # (3,)
print("x.shape =", x.shape)   # (4, 3)
print("b.shape =", b.shape)   # (1,)
print("y.shape =", y.shape)   # (4, 3)  ← 广播结果
print("y.grad_fn =", y.grad_fn)

y.sum().backward()





print("w.grad =", w.grad)
print("b.grad =", b.grad)

y.backward()           # 反向：求 dy/dw 和 dy/db

print("\nw.grad =", w.grad, "  (dy/dw = x = 3)")
print("b.grad =", b.grad, "  (dy/db = 1)")


sec("练习五")
