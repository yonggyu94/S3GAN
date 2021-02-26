from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
import torch.utils.data as data
import torch

from PIL import Image
import os
import random


def celeba_data_loader(root, batch_size):
    transform =  Transforms.Compose([Transforms.CenterCrop((120, 120)),
                                     Transforms.Resize((84, 84)),
                                     Transforms.RandomCrop((64, 64)),
                                     Transforms.ToTensor(),
                                     Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                          std=(0.5, 0.5, 0.5))])

    dataset = ImageFolder(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    cls_num = len(os.listdir(root))

    return dataloader, len(dataset), cls_num


def vgg_data_loader(root, batch_size, resize_size, crop_size):
    transform_list = []

    transform_list += [Transforms.Resize((resize_size, resize_size))]
    transform_list += [Transforms.RandomCrop((crop_size, crop_size))]
    # PIL -> Tensor
    transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transform_list)

    dataset = ImageFolder(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    cls_num = len(os.listdir(root))

    return dataloader, len(dataset), cls_num


def omniglot_data_loader(root, batch_size, resize_size, crop_size):
    omniglot_transformer = Transforms.Compose([Transforms.Resize((resize_size, resize_size)),
                                               Transforms.RandomCrop((crop_size, crop_size)),
                                               Transforms.Grayscale(num_output_channels=1),
                                               Transforms.ToTensor(),
                                               Transforms.Normalize(mean=[0.5],
                                                                    std=[-0.5])])

    dataset = ImageFolder(root, transform=omniglot_transformer)
    cls_num = len(os.listdir(root))

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader, len(dataset), cls_num


def img_dataloder(args, root):
    train_dataset = ImageDataset(args, root)
    dataloder = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    return dataloder, len(train_dataset)


class BASE(data.Dataset):
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

    def _read_image(self, img_path, color_type='RGB'):
        return Image.open(img_path).convert(color_type)

    def omniglot_transformer(self):
        transformer = Transforms.Compose([Transforms.Resize((self.args.resize_size,
                                                            self.args.resize_size)),
                                          Transforms.RandomCrop((self.args.crop_size,
                                                                 self.args.crop_size)),
                                          Transforms.Grayscale(num_output_channels=1),
                                          Transforms.ToTensor(),
                                          Transforms.Normalize((0.5,), (-0.5,))])
        return transformer

    def vgg_transformer(self):
        transformer = Transforms.Compose([Transforms.Resize((self.args.resize_size,
                                                            self.args.resize_size)),
                                          Transforms.RandomCrop((self.args.crop_size,
                                                                 self.args.crop_size)),
                                          Transforms.ToTensor(),
                                          Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))])
        return transformer
    
    def celeba_transformer(self):
        transformer = Transforms.Compose([Transforms.Resize((self.args.resize_size,
                                                            self.args.resize_size)),
                                          Transforms.RandomCrop((self.args.crop_size,
                                                                 self.args.crop_size)),
                                          Transforms.ToTensor(),
                                          Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))])
        return transformer
    
    def animal_transformer(self):
        transformer = Transforms.Compose([Transforms.Resize((self.args.resize_size,
                                                            self.args.resize_size)),
                                          Transforms.RandomCrop((self.args.crop_size,
                                                                 self.args.crop_size)),
                                          Transforms.ToTensor(),
                                          Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))])
        return transformer
    
    def cub_transformer(self):
        transformer = Transforms.Compose([Transforms.Resize((self.args.resize_size,
                                                            self.args.resize_size)),
                                          Transforms.RandomCrop((self.args.crop_size,
                                                                 self.args.crop_size)),
                                          Transforms.ToTensor(),
                                          Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))])
        return transformer
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class ImageDataset(BASE):
    def __init__(self, args, data_path):
        super(ImageDataset, self).__init__(args)
        self.class_list = sorted(os.listdir(data_path))
        self.class_path_list = []
        self.img_paths = []
        self.labels = []
        self.str_list = sorted(list(map(str, range(len(self.class_list)))))
        self.dataset = args.dataset
        
        for _class in self.class_list:
            class_path = os.path.join(data_path, _class)
            img_list = sorted(os.listdir(class_path))

            self.class_path_list.append(class_path)

            for img in img_list:
                img_path = os.path.join(class_path, img)
                self.img_paths.append(img_path)
            self.labels.append(_class)
        
        self.labels = sorted(self.labels)
        if args.dataset == "vgg":
            self.transformer = self.vgg_transformer()
        elif args.dataset == "omniglot":
            self.transformer = self.omniglot_transformer()
        elif args.dataset == "celeba":
            pass
        elif args.dataset == "animal":
            self.transformer = self.animal_transformer()
            print(128)
        elif args.dataset == "cub":
            self.transformer = self.vgg_transformer()
            print(64)
            pass

    def __getitem__(self, index):
        img = self._read_image(self.img_paths[index], color_type='RGB')
        img = self.transformer(img)
        label = self.img_paths[index].split("/")[-2]
        
        if self.img_paths[index].split("/")[-1].split("_")[0] == "unlabeled":
            return img, torch.tensor(len(self.labels))
        else:
            return img, torch.tensor(self.labels.index(label))

    def __len__(self):
        return len(self.img_paths)

