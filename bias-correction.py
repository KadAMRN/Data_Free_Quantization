import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import optim, cuda
from torchvision import transforms, datasets, models
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchinfo import summary
import torchmetrics
import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

batch_size = 32

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

#If you use multiple gpus it turns statement multi_gpu = True. Probably useful for large datasets
# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Using the function with the os module
create_folder_if_not_exists("data_")

train_data = datasets.MNIST(
    root="data_",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.MNIST(
    root="data_",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

print(len(train_data), len(test_data))

class_names = train_data.classes
print(class_names)

image = train_data[0][0]
plt.imshow(image.squeeze(), cmap="gray")
plt.axis(False)
image.shape

from torch.utils.data import DataLoader
BATCH_SIZE = 32
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),                # Resize to 224x224
    transforms.ToTensor(),
    # Include normalization if necessary, for example:
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data.transform = transform
test_data.transform = transform

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False) 

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, len(class_names))
model.cuda()

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

summary(model)

if train_on_gpu:
    model = model.to('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test_step(model,
              data_loader,
              loss_fn,
              optimizer,
              accuracy_fn,
              device=device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X,y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss
            acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            test_acc += acc
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_pred_tensor = torch.cat(y_preds)

len(y_pred_tensor)
y_pred_tensor.shape
test_data.targets.shape, len(y_pred_tensor.unique())

confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10,7)
)

print(test_data[0][0].shape)

# import torch
# import matplotlib.pyplot as plt

# plt.figure(figsize=(20, 20))
# count = 1

# for i, y in enumerate(y_pred_tensor):
#     if count > 25:
#         plt.figure(figsize=(20, 20))
#         count = 1

#     if y != test_data.targets[i]:
#         plt.subplot(5, 5, count)
#         plt.axis(False)
#         image = test_data[i][0]
#         grayscale = image.mean(dim=0, keepdim=True)
        
#         # Replicate the grayscale channel to create a 3-channel grayscale image
#         grayscale_3channel = grayscale.repeat(3, 1, 1)

#         # Normalize the image if it's not in the range [0, 1]
#         if grayscale_3channel.max() > 1:
#             grayscale_3channel = grayscale_3channel / 255

#         plt.imshow(grayscale_3channel.permute(1, 2, 0), cmap='gray')
#         predStr = f"Pred: {class_names[y]}"
#         targetStr = f"Actual: {class_names[test_data.targets[i]]}"
#         plt.title(predStr, color='red')
#         plt.text(.5, 1, targetStr, color='green')

#     count += 1  # Increment count outside the conditional block
print("Ok_1")

#Evaluate Model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

original_model = model
original_accuracy = evaluate_model(original_model, test_dataloader)








# Assuming we are using 8-bit quantization
N_levels = 256
zero_point = 0

# 2. Quantization and De-quantization Functions
def quantize(tensor, scale):
    tensor_int = torch.round(tensor / scale) + zero_point
    tensor_quantized = torch.clamp(tensor_int, 0, N_levels - 1)
    return tensor_quantized

def dequantize(tensor_quantized, scale):
    tensor_float = (tensor_quantized - zero_point) * scale
    return tensor_float

# 3. Model Adaptation
# for every layer.
class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QuantizedConv2d, self).__init__(*args, **kwargs)
        # Define scale for each layer based on the pre-trained weights
        self.scale = torch.std(self.weight) / (N_levels / 2) # This is a simplification

    def forward(self, x):
        # Quantize the input
        x_quantized = quantize(x, self.scale)
        # Perform the convolution on quantized input
        x_conv = F.conv2d(x_quantized, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        # Dequantize the output
        x_dequantized = dequantize(x_conv, self.scale)
        return x_dequantized

# Replace the original Conv2d layers with QuantizedConv2d
def replace_conv_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Create a new layer and copy properties from the existing layer
            new_layer = QuantizedConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                        stride=module.stride, padding=module.padding, dilation=module.dilation, 
                                        groups=module.groups, bias=(module.bias is not None))
            # Copy the weights from the old to new layer
            new_layer.load_state_dict(module.state_dict())
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Conv2d):
                    new_layer = QuantizedConv2d(child_module.in_channels, child_module.out_channels, 
                                                child_module.kernel_size, stride=child_module.stride, 
                                                padding=child_module.padding, dilation=child_module.dilation, 
                                                groups=child_module.groups, bias=(child_module.bias is not None))
                    new_layer.load_state_dict(child_module.state_dict())
                    module[child_name] = new_layer
        else:
            replace_conv_layers(module)

# Replace the layers in the model
quantized_model = model 
replace_conv_layers(quantized_model.features)


# Evaluate the quantized model
quantized_accuracy = evaluate_model(quantized_model, test_loader)

# Now plot the results
labels = ['Original', 'Quantized']
accuracies = [original_accuracy, quantized_accuracy]

fig, ax = plt.subplots()
ax.bar(labels, accuracies, color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Before and After Quantization')
ax.set_ylim([0, 1])  # Assuming accuracy is a proportion between 0 and 1

# Display the actual accuracy on top of each bar
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', color='black')

fig.savefig('accuracy_comparison.png', bbox_inches='tight')
plt.show()
