import sys, time, random
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
try:
    import cv2
    print("OpenCV:", cv2.__version__)
except ImportError:
    cv2 = None
    print("OpenCV not found (only needed for the optional photo test)")
SEED = 42 # fixed seed so your results are repeatable
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__)
print("Training on:", device)


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
train_full = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
# 45,000 for training, 5,000 for validation
train_set, val_set = random_split(
train_full, [45000, 5000], generator=torch.Generator().manual_seed(SEED))
BATCH = 128
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=2)
print("train / val / test sizes:", len(train_set), len(val_set), len(test_set))


raw = torchvision.datasets.CIFAR10("./data", train=True, download=False) # PIL images
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, i in zip(axes.flat, range(16)):
    img, label = raw[i]
    ax.imshow(img)
    ax.set_title(CLASSES[label], fontsize=9)
    ax.axis("off")
plt.suptitle("What the camera sees: 32x32 real photographs")
plt.show()
print("Images per class:", dict(zip(CLASSES, np.bincount(raw.targets))))
images, labels = next(iter(train_loader))
print("images:", images.shape, "| labels:", labels.shape)
print("pixel range after normalisation: %.2f to %.2f" % (images.min(), images.max()))
class SceneCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes), # raw scores (logits), no softmax here
        )
    def forward(self, x):
        return self.classifier(self.features(x))
model = SceneCNN().to(device)
print(model)
x = torch.randn(1, 3, 32, 32).to(device) # one fake 32x32 colour image
for layer in model.features:
    x = layer(x)
    print(f"{layer.__class__.__name__:<10} -> {tuple(x.shape)}")
print("Flattened length fed to the classifier:", x.numel())
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1024), nn.ReLU(), nn.Linear(1024, 10))
print(f"SceneCNN parameters : {count_params(model):>10,}")
print(f"Plain MLP parameters: {count_params(mlp):>10,}")
print()
for name, p in model.named_parameters():
    print(f"{name:<22}{str(tuple(p.shape)):<20}{p.numel():>9,}")
layers = [("conv1", 3, 1), ("pool1", 2, 2), ("conv2", 3, 1),
          ("pool2", 2, 2), ("conv3", 3, 1), ("pool3", 2, 2)] # (name, kernel, stride)
r, j = 1, 1
for name, k, s in layers:
    r = r + (k - 1) * j
    j = j * s
print(f"{name}: each neuron sees a {r} x {r} patch of the original image")
criterion = nn.CrossEntropyLoss()
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)
model.zero_grad()
loss = criterion(model(images), labels) # forward pass + loss
loss.backward() # backward pass (backpropagation)
print(f"Initial loss: {loss.item():.3f} (guessing = ln(10) = {np.log(10):.3f})")
for name, p in model.named_parameters():
    if "weight" in name:
        print(f"{name:<22} gradient norm = {p.grad.norm().item():.4f}")
EPOCHS = 8 if device.type != "cpu" else 4 # fewer epochs on CPU
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x) # forward pass
            loss = criterion(out, y) # cross-entropy loss
            if train:
                optimizer.zero_grad()
                loss.backward() # backpropagation
                optimizer.step() # gradient descent step
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, ta = run_epoch(train_loader, train=True)
    vl, va = run_epoch(val_loader, train=False)
    for k, v in zip(history, (tl, vl, ta, va)):
        history[k].append(v)
    print(f"Epoch {epoch}/{EPOCHS} | train loss {tl:.3f} acc {ta:.1%} "
          f"| val loss {vl:.3f} acc {va:.1%} | {time.time() - t0:.0f}s")
epochs = range(1, len(history["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(epochs, history["train_loss"], "o-", label="train")
ax[0].plot(epochs, history["val_loss"], "o-", label="validation")
ax[0].set_title("Loss (lower is better)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(epochs, history["train_acc"], "o-", label="train")
ax[1].plot(epochs, history["val_acc"], "o-", label="validation")
ax[1].set_title("Accuracy (higher is better)"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()


def predict_photo(path, topk=3):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(h, w) # centre square crop
    rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    x = T.Normalize(MEAN, STD)(x).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        p_ = F.softmax(model(x), dim=1)[0].cpu()
        top = p_.topk(topk)
    plt.imshow(cv2.resize(rgb, (256, 256))); plt.axis("off"); plt.show()
    for score, i in zip(top.values, top.indices):
        print(f"{CLASSES[i]:<11}{score:.1%}")




@torch.no_grad()
def predict_all(loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(probs), torch.cat(ys)

probs, ys = predict_all(test_loader)
preds = probs.argmax(1)

conf, _ = probs.max(1)
wrong = (preds != ys).nonzero().flatten()
worst = wrong[conf[wrong].argsort(descending=True)[:8]]
raw_test = torchvision.datasets.CIFAR10("./data", train=False, download=False)
fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for ax, idx in zip(axes, worst.tolist()):
    img, _ = raw_test[idx]
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"true: {CLASSES[ys[idx]]}\npred: {CLASSES[preds[idx]]}\n({conf[idx]:.0%} sure)", fontsize=8)
plt.show()



VEHICLES = torch.tensor([0, 1, 8, 9]) # airplane, automobile, ship, truck
true_vehicle = torch.isin(ys, VEHICLES)
pred_vehicle = torch.isin(preds, VEHICLES)
print(f"10-class (object-level) accuracy : {(preds == ys).float().mean():.2%}")
print(f"Vehicle vs animal (scene-level) : {(true_vehicle == pred_vehicle).float().mean():.2%}")



def out_size(h, k, p, s):
    return (h + 2 * p - k) // s + 1

for k, p_, s in [(3, 0, 1), (3, 0, 2), (3, 1, 1), (3, 1, 2)]:
    real = nn.Conv2d(3, 8, kernel_size=k, padding=p_, stride=s)(torch.randn(1, 3, 32, 32))
    print(f"kernel={k} padding={p_} stride={s}: formula {out_size(32, k, p_, s)}"
          f" | PyTorch {real.shape[-1]}")


sub_loader = DataLoader(Subset(train_set, range(10000)), batch_size=BATCH,
                        shuffle=True, num_workers=2)

def quick_lr_test(lr, epochs=2):
    torch.manual_seed(SEED)
    m = SceneCNN().to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        m.train()
        for x, y in sub_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()
    m.eval(); correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            correct += (m(x.to(device)).argmax(1).cpu() == y).sum().item()
    return loss.item(), correct / len(val_set)
  

def predict_photo(path, topk=3):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(h, w) # centre square crop
    rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    x = T.Normalize(MEAN, STD)(x).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        p_ = F.softmax(model(x), dim=1)[0].cpu()
        top = p_.topk(topk)
    plt.imshow(cv2.resize(rgb, (256, 256))); plt.axis("off"); plt.show()
    for score, i in zip(top.values, top.indices):
        print(f"{CLASSES[i]:<11}{score:.1%}")

predict_photo("/content/drive/MyDrive/kitty.jpg")
