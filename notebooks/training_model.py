import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# ============================
# 0. Архитектура сети (Net)
# ============================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # после двух пуллингов с шагом 2:
        # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)           # 32x32 -> 16x16

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)           # 16x16 -> 8x8

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten: [batch, 128*8*8]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================
# 1. Гиперпараметры
# ============================
batch_size = 128
num_epochs = 15
learning_rate = 0.001
model_path = "./artifacts/cifar_net.pth"

# создаём папку для модели, если её нет
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# ============================
# 2. Датасет CIFAR-10
# ============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True,
    download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size,
    shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False,
    download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size,
    shuffle=False, num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ============================
# 3. Модель, лосс, оптимизатор
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ============================
# 4. Обучение + валидация
# ============================
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    running_loss = 0.0

    # --- обучение ---
    net.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"[{epoch+1}, {i+1:4d}] loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # --- проверка на тесте ---
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)
    acc = 100.0 * correct / total
    print(f"Test loss: {avg_test_loss:.4f} | Test accuracy: {acc:.2f}%")

print("\nFinished Training")

# ============================
# 5. Сохранение модели
# ============================
torch.save(net.state_dict(), model_path)
print(f"Model saved to {model_path}")
