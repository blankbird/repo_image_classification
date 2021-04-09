import os
from torchvision import transforms, datasets
#import torch.utils.data as data
import torch
import torchvision.models.inception

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
a = train_dataset.classes   # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
b = train_dataset.class_to_idx  # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
c = train_dataset.imgs  # List of (image path, class_index) tuples
#print(a),print(b),print(c)
#print(train_dataset)  # 抽象类

train_loader = torch.utils.data.dataloader.DataLoader(train_dataset,batch_size = 32,shuffle = True,num_workers = 4)
print(len(train_loader))