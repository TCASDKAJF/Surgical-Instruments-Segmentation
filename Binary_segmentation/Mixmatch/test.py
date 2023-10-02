import numpy as np
import torch
import sys
import torchvision
import torchvision.transforms as transforms
import torchvision.models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import matplotlib.pyplot as plt
import utils

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

weight_file = sys.argv[1]

model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT', progress=True)
model.classifier = DeepLabHead(2048, 3)
model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
if torch.cuda.is_available():
   model.to(device)

model.eval()
criterion = torch.nn.CrossEntropyLoss()

# WARINGIN: !!!!!!!!!! 
# resize to 128x128 if run out of memory
# !!!!!!!!!!
transform_test = transforms.Compose([transforms.Resize((256,256)), 
                  #transforms.CenterCrop(256), 
                  transforms.ToTensor(), 
                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


transform_Mask = transforms.Compose([transforms.Resize((256,256)), 
                  #transforms.CenterCrop(256), 
                  transforms.PILToTensor(), 
                  transforms.Lambda(lambda x: np.asarray(x).astype('int32') - 1)])

transform_raw = transforms.Compose([transforms.Resize((256,256)), 
                    #transforms.CenterCrop(256), 
                    transforms.ToTensor()])

Test_DataSet = torchvision.datasets.OxfordIIITPet(root='./data', split= 'test',target_types='segmentation' ,download=True, transform=transform_test, target_transform=transform_Mask)
Raw_DataSet = torchvision.datasets.OxfordIIITPet(root='./data', split= 'test',target_types='segmentation' ,download=True, transform=transform_raw, target_transform=transform_Mask)
test_loader = torch.utils.data.DataLoader(Test_DataSet, batch_size=10, shuffle=False, num_workers=2)
raw_loader = torch.utils.data.DataLoader(Raw_DataSet, batch_size=10, shuffle=False, num_workers=2)

Test_loss = 0.0
Test_acc = 0.0
Test_MIoU = 0.0

for i, Test_Data in enumerate(test_loader, 0):
  Test_imgs, Test_labels = Test_Data
  Test_labels = utils.transform_mask(Test_labels.squeeze(1))

  if torch.cuda.is_available():
    Test_imgs = Test_imgs.to(device)
    Test_labels = Test_labels.to(device)
  predictions = model(Test_imgs)['out']
  predictions = torch.softmax(predictions, 1)

  # calculate losses
  loss = criterion(predictions, Test_labels.float())
  Test_loss += loss.item()
  predictions = torch.argmax(predictions, 1)
  Test_labels = torch.argmax(Test_labels, 1)
  Test_acc += utils.metrics(predictions, Test_labels)
  Test_MIoU += utils.MIoU(predictions, Test_labels)
  
Test_loss = Test_loss / len(test_loader)
Test_acc = Test_acc / len(test_loader)
Test_MIoU = Test_MIoU / len(test_loader)
print("Test set loss: %.5f" %(Test_loss))
print("Test set acc: %.5f" %(Test_acc))
print("Test set MIoU: %.5f" %(Test_MIoU))

