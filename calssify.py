import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

data_transforms_inception = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
}


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
}


data_dir = 'spectrograms'

image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
image_dataset_inception = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms_inception['train'])

train_size = int(0.8 * len(image_dataset))
val_size = len(image_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size])

train_size_inc = int(0.8 * len(image_dataset_inception))
val_size_inc = len(image_dataset_inception) - train_size_inc
train_dataset_inc, val_dataset_inc = torch.utils.data.random_split(image_dataset_inception, [train_size_inc, val_size_inc])

dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
}

dataloaders_inception = {
    'train': torch.utils.data.DataLoader(train_dataset_inc, batch_size=32, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(val_dataset_inc, batch_size=32, shuffle=False, num_workers=0)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
dataset_sizes_inception = {'train': len(train_dataset_inc), 'val': len(val_dataset_inc)}
class_names = image_dataset.classes

print(f"类别名称：{class_names}")
print(f"训练集样本数：{dataset_sizes['train']}, 验证集样本数：{dataset_sizes['val']}")

def train_and_save_model(model, model_name, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=10):
    print(f"开始训练模型：{model_name}")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

  
                with torch.set_grad_enabled(phase == 'train'):
                    if model_name == 'InceptionV3' and phase == 'train':
            
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2  
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().numpy())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f' {model_name} 最佳验证集准确率: {best_acc:.4f}')


    model.load_state_dict(best_model_wts)


    result_dir = 'training_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, f'{model_name}_training_results.txt')
    with open(result_file, 'w') as f:
        f.write('Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n')
        for i in range(num_epochs):
            f.write(f"{i+1}\t{train_loss_history[i]:.4f}\t{val_loss_history[i]:.4f}\t{train_acc_history[i]:.4f}\t{val_acc_history[i]:.4f}\n")

    print(f'模型 {model_name} 的训练结果已保存到 {result_file}')
    return model


models_to_train = []


model_resnet18 = models.resnet18(pretrained=True)
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, len(class_names))
model_resnet18 = model_resnet18.to(device)
models_to_train.append(('ResNet18', model_resnet18))


model_vgg16 = models.vgg16(pretrained=True)
num_ftrs = model_vgg16.classifier[6].in_features
model_vgg16.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model_vgg16 = model_vgg16.to(device)
models_to_train.append(('VGG16', model_vgg16))


model_densenet121 = models.densenet121(pretrained=True)
num_ftrs = model_densenet121.classifier.in_features
model_densenet121.classifier = nn.Linear(num_ftrs, len(class_names))
model_densenet121 = model_densenet121.to(device)
models_to_train.append(('DenseNet121', model_densenet121))


model_inception_v3 = models.inception_v3(pretrained=True, aux_logits=True)
num_ftrs = model_inception_v3.fc.in_features
model_inception_v3.fc = nn.Linear(num_ftrs, len(class_names))


num_ftrs_aux = model_inception_v3.AuxLogits.fc.in_features
model_inception_v3.AuxLogits.fc = nn.Linear(num_ftrs_aux, len(class_names))
model_inception_v3 = model_inception_v3.to(device)
models_to_train.append(('InceptionV3', model_inception_v3))

criterion = nn.CrossEntropyLoss()


for model_name, model in models_to_train:

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    if model_name == 'InceptionV3':
        current_dataloaders = dataloaders_inception
        current_dataset_sizes = dataset_sizes_inception
    else:
        current_dataloaders = dataloaders
        current_dataset_sizes = dataset_sizes


    model = train_and_save_model(model, model_name, criterion, optimizer, exp_lr_scheduler,
                                 current_dataloaders, current_dataset_sizes, device, num_epochs=10)


    model_save_path = os.path.join('trained_models', f'{model_name}.pth')
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    torch.save(model.state_dict(), model_save_path)
    print(f'模型 {model_name} 已保存到 {model_save_path}')


class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model.eval()
        self.cuda = use_cuda

        if self.cuda:
            self.model = model.cuda()

        self.feature = None
        self.gradient = None
        self.target_layer_names = target_layer_names

        def save_feature(module, input, output):
            self.feature = output.detach()

        def save_gradient(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(save_feature)
                module.register_backward_hook(save_gradient)

    def __call__(self, input_tensor, index=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0] 

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature, dim=1)

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam[0]
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_cam_on_image(img, mask, output_dir, model_name, img_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap * 0.4 + img * 0.6
    plt.figure(figsize=(6, 6))
    plt.imshow(cam.astype('uint8'))
    plt.axis('off')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{model_name}_{img_name}_gradcam.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_gradcam_visualizations(models_list, device):

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    preprocess_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    num_images = 5  
    for model_name, model in models_list:
        print(f"生成模型 {model_name} 的 Grad-CAM 可视化结果...")

        model_load_path = os.path.join('trained_models', f'{model_name}.pth')
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.eval()


        if model_name == 'InceptionV3':
            preprocess_func = preprocess_inception
            dataloader = dataloaders_inception['val']
        else:
            preprocess_func = preprocess
            dataloader = dataloaders['val']

 
        if model_name == 'ResNet18':
            target_layer_names = ['layer4']
        elif model_name == 'VGG16':
            target_layer_names = ['features.29']
        elif model_name == 'DenseNet121':
            target_layer_names = ['features.denseblock4.denselayer16']
        elif model_name == 'InceptionV3':
            target_layer_names = ['Mixed_7c']

        grad_cam = GradCAM(model=model, target_layer_names=target_layer_names, use_cuda=False)


        count = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            for i in range(inputs.size(0)):
                if count >= num_images:
                    break

                input_img = inputs[i].unsqueeze(0)
                img_np = input_img.cpu().numpy()[0].transpose((1, 2, 0))
                img_np = (img_np * 0.5) + 0.5  # 反归一化
                img_np = np.clip(img_np * 255, 0, 255).astype('uint8')

      
                mask = grad_cam(input_img)

 
                output_dir = 'gradcam_results'
                img_name = f'image_{count}'
                show_cam_on_image(img_np, mask, output_dir, model_name, img_name)
                count += 1

            if count >= num_images:
                break

generate_gradcam_visualizations(models_to_train, device)