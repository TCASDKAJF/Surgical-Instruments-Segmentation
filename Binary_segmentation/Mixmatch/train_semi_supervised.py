import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset
import os

import utils
import mixmatch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

num_epoch = 30
batch_size = 8
train_val_ = 0.8 
test_ = 0.2
train_ = 0.9     
val_ = 0.1       
label_ = 0.7   
unlabel_ = 0.3  
K = 2
T = 0.5
alpha = 0.3
unlabelled_weight = 10
L = 3

transform1 = transforms.Compose([transforms.Resize((256,256)),
                  transforms.ToTensor(), 
                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

def label_transform(x):
    return (np.asarray(x) / 255).astype('int32')

transform2 = transforms.Compose([transforms.Resize((256,256)),
                                 transforms.PILToTensor(),
                                 transforms.Lambda(label_transform)])

transform3 = transforms.Compose([transforms.Resize((256,256)),
                    transforms.ToTensor()])

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = os.listdir(os.path.join(root, 'JPEGImages'))
        self.label_list = os.listdir(os.path.join(root, 'SegmentationClass'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'JPEGImages', self.image_list[idx])
        label_path = os.path.join(self.root, 'SegmentationClass', self.label_list[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

root_path = r'./VOCdevkit/VOC2007'

DataSet = CustomDataset(root=root_path, transform=transform3, target_transform=transform2)

dev_size = int(train_val_ * len(DataSet))
test_size = len(DataSet) - dev_size
dev_set, test_set = torch.utils.data.random_split(DataSet, [dev_size, test_size])

train_size = int(train_ * len(dev_set))
val_size = len(dev_set) - train_size
train_set, val_set = torch.utils.data.random_split(dev_set, [train_size, val_size])

labeled_size = int(label_ * len(train_set))
unlabeled_size = len(train_set) - labeled_size
labeled_set, unlabeled_set = torch.utils.data.random_split(train_set, [labeled_size, unlabeled_size])


print('Number of data in the labeled: ', len(labeled_set))
print('Number of data in the unlabeled: ', len(unlabeled_set))
print('Number of data in the val set: ', len(val_set))
print('Number of data in the test set: ', len(test_set))

# DataLoader
label_loader = torch.utils.data.DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
unlabelled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

# Train semi-supervised model
model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT', progress=True)
model.classifier = DeepLabHead(2048, 2)
if torch.cuda.is_available():
   model.to(device)
Loss_CrossEntropy = torch.nn.CrossEntropyLoss()
Loss_L2 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

plot_running_loss = []
plot_val_loss = []
plot_trainging_acc = []
plot_val_acc = []
plot_training_MIoU = []
plot_val_MIoU = []
transform = transforms.Compose([transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
# TRAINGING AND VALIDATION
for epoch in range(num_epoch):
    print('Epoch {}/{}'.format(epoch+1, num_epoch))
    model.train()
    running_losses = 0
    running_acc = 0
    val_losses = 0
    val_acc = 0
    running_MIoU = 0
    val_MIoU = 0

    num_training = max(len(label_loader), len(unlabelled_loader))
    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabelled_loader)
    val_iter = iter(val_loader)
    
    for i in range(num_training):
        model.train()
        # Load labelled Data
        try:
          input_labelled, labelled_mask = next(label_iter)
        except:
          label_loader = torch.utils.data.DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
          label_iter = iter(label_loader)
          input_labelled, labelled_mask = next(label_iter)
        
        try:
          input_unlabelled, _ = next(unlabel_iter)
        except:
          unlabelled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
          unlabel_iter = iter(unlabelled_loader)
          input_unlabelled, _ = next(unlabel_iter)
        
        model.eval()
        X, U, p, q = mixmatch.MixMatch(model, input_labelled, input_unlabelled, labelled_mask.squeeze(1), T, K, alpha)

        model.train()          
        optimizer.zero_grad()
        # forward
        out_x = model(X)['out']
        out_u = model(U)['out']
        out_x = torch.softmax(out_x, 1)
        out_u = torch.softmax(out_u, 1)

        # Calculate loss
        loss_label_x = Loss_CrossEntropy(out_x, p.float()) 
        # loss_label_u = Loss_L2(out_u, q.float()) / L
        loss_label_u = Loss_CrossEntropy(out_u, q.float())  
        loss = 0.5 * loss_label_x + 0.5 * loss_label_u
        # loss = loss_label_x + unlabelled_weight * loss_label_u
        
        # backword
        loss.backward()
        optimizer.step()
        running_losses += loss.item()

        #training acc
        out_x_acc = torch.argmax(out_x, 1)
        labelled_mask_acc = labelled_mask.squeeze(1)
        labelled_mask_acc = labelled_mask_acc.to(device)
        running_acc += utils.metrics(out_x_acc, labelled_mask_acc)
        running_MIoU += utils.MIoU(out_x_acc, labelled_mask_acc)
        
    plot_running_loss.append(running_losses / num_training)
    plot_trainging_acc.append(running_acc / num_training)
    plot_training_MIoU.append(running_MIoU / num_training)
    print('Training loss: %.5f' % (running_losses / num_training))

  # Validation
    model.eval()
    with torch.no_grad():
        for j, Val_data in enumerate(val_loader, 0):
            # get the inputs: data is a list of [inputs, labels]
            Val_inputs, Val_labels = Val_data
            Val_inputs = transform(Val_inputs)
            Val_labels = utils.transform_mask(Val_labels.squeeze(1))
            Val_inputs = Variable(Val_inputs)
            Val_labels = Variable(Val_labels)
            if torch.cuda.is_available():
                Val_inputs = Val_inputs.to(device)
                Val_labels = Val_labels.to(device)
            predictions = model(Val_inputs)['out']
            predictions = torch.softmax(predictions, 1)
            # calculate losses
            val_loss = Loss_CrossEntropy(predictions, Val_labels.float())
            val_losses += val_loss.item()

            #validation acc
            predictions_acc = torch.argmax(predictions, 1)
            val_mask_acc = torch.argmax(Val_labels, 1)
            val_acc += utils.metrics(predictions_acc, val_mask_acc)
            val_MIoU += utils.MIoU(predictions_acc, val_mask_acc)
    
    plot_val_loss.append(val_losses / len(val_loader))
    plot_val_acc.append(val_acc / len(val_loader))
    plot_val_MIoU.append(val_MIoU / len(val_loader))
    print('Validation loss: %.5f' % (val_losses / len(val_loader)))
        
torch.save(model.state_dict(), 'semi_supervised.pt')

# plotting
num_loss = len(np.array(plot_running_loss))
plt.plot(np.arange(num_loss)+1, np.array(plot_running_loss), 'b-', np.arange(num_loss)+1, np.array(plot_val_loss), 'r-', linewidth=2)
plt.legend(labels=['training','validation'],loc='best')
plt.xlabel("number of epoch")
plt.ylabel("loss")
plt.title("semi-supervised losses vs epoches")
plt.savefig('semi_supervised_loss.png')

plt.plot(np.arange(num_loss)+1, np.array(plot_trainging_acc), 'b-', np.arange(num_loss)+1, np.array(plot_val_acc), 'r-', linewidth=2)
plt.legend(labels=['training','validation'],loc='best')
plt.xlabel("number of epoch")
plt.ylabel("accuracy")
plt.title("semi-supervised accuracy vs epoches")
plt.savefig('semi_supervised_acc.png')

plt.plot(np.arange(num_loss)+1, np.array(plot_training_MIoU), 'b-', np.arange(num_loss)+1, np.array(plot_val_MIoU), 'r-', linewidth=2)
plt.legend(labels=['training','validation'],loc='best')
plt.xlabel("number of epoch")
plt.ylabel("MIoU")
plt.title("semi-supervised MIoU vs epoches")
plt.savefig('semi_supervised_MIoU.png')