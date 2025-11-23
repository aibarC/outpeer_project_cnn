import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)          # 32x32 -> 16x16

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)           # 16x16 -> 8x8

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten: [batch, 128*8*8]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("artifacts/cifar_net.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Загрузи изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Загруженное изображение", use_column_width=True)

    # предобработка
    x = transform(img).unsqueeze(0)  # [1, 3, 32, 32]

    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        label = CLASSES[predicted.item()]

    st.markdown(f"### Предсказанный класс: **{label}**")
