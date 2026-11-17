import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1=nn.Linear(1,8)
        self.act=nn.ReLU()
        self.fc2=nn.Linear(8,1)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return x

x_train = torch.linspace(-1, 1, 101).unsqueeze(1)
y_train = 2 * x_train + 1

model=TinyNet()

loss_fn=nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03
)

for epoch in range(401):
    model.train()

    # ① Forward：模型進行預測
    pred = model(x_train)

    # ② Loss：計算預測錯誤
    loss = loss_fn(pred, y_train)

    # ③ 清除上一輪梯度
    optimizer.zero_grad()

    # ④ Backward：計算梯度
    loss.backward()

    # ⑤ Optimizer：更新模型參數
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

model.eval()

with torch.no_grad():
    x=torch.tensor([[0.5]]) 
    y=model(x)
    print(y.item()) # 2*0.5+1
