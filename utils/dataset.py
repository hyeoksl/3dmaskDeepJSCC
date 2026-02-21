import os
import io
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG
from torch.utils.data import DataLoader

__all__ = ['loadData', 'load_dataloaders']

DATA_ROOT = os.environ.get('DATASET_ROOT', '/datasets')

class addContinuousNoise(object):
    def __init__(self, image_size, clip_range, quantization_step):
        assert isinstance(image_size, tuple)
        assert isinstance(clip_range, tuple)
        assert isinstance(quantization_step, (int, float))
        self.image_size = image_size
        self.clip_range = clip_range
        self.quantization_step = quantization_step

    def __call__(self, image):
        noise = torch.rand(self.image_size) * self.quantization_step - (self.quantization_step/2)
        image.add_(noise)
        image.clamp_(min=self.clip_range[0], max=self.clip_range[1])
        return image
    
class indexedCachedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images, transform=None, **kwargs):
        self.num_images = num_images
        self.transform = transform
        
        self.images = []
        for i in range(num_images):
            img_name = fname_format.format(i+1)
            img_file = os.path.join(root_dir, img_name)
            image = Image.open(img_file)
            image = image.convert("RGB")
            self.images.append(image)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
class indexedCachedRepeatedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images_per_ds, num_reps, transform=None, **kwargs):
        self.num_images_per_ds = num_images_per_ds
        self.num_reps = num_reps
        self.transform = transform
        
        self.images = []
        for i in range(num_images_per_ds):
            img_name = fname_format.format(i+1)
            img_file = os.path.join(root_dir, img_name)
            image = Image.open(img_file)
            image = image.convert("RGB")
            self.images.append(image)

    def __len__(self):
        return self.num_images_per_ds * self.num_reps

    def __getitem__(self, idx):
        idx %= self.num_images_per_ds
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
class indexedCodeCachedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images, transform=None, **kwargs):
        self.num_images = num_images
        self.transform = transform
        
        self.image_bytes = []
        for i in range(num_images):
            img_name = fname_format.format(i+1)
            img_file = os.path.join(root_dir, img_name)
            with open (img_file, 'rb') as f:
                img_bytes = f.read()
            self.image_bytes.append(img_bytes)
        
        self.jpeg = TurboJPEG()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_bytes = self.image_bytes[idx]
        try:
            image = self.jpeg.decode(img_bytes)
            image = Image.fromarray(image)
        except Exception as e: # 예외 구문 구체화
            image = Image.open(io.BytesIO(img_bytes))
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image
    
def loadTorchDatasets(data_config):
    patch_size = data_config.pop('patch_size')
    transform_type = data_config.pop('transform_type')
    if transform_type == 'train':
        transform = v2.Compose([
            v2.RandomResize(640, 1200),
            v2.RandomCrop(patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            addContinuousNoise(
                image_size=patch_size,
                clip_range=(0, 1),
                quantization_step=1/256
            )
        ])
    elif transform_type == 'test': # TODO : 256 cropping!!
        transform = torchvision.transforms.Compose([
            v2.RandomCrop(patch_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    elif transform_type == 'valid':
        transform = torchvision.transforms.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else:
        raise NameError(f'transform_type {data_config["transform_type"]} not supported')
    
    dataset_class = data_config.pop('dataset_class')
    data_config['transform'] = transform
    
    dataset = dataset_class(**data_config)
    return dataset


def loadDatasetImages(data_config):
    image_list = []
    for i in range(data_config['num_images']):
        img_name = data_config['fname_format'].format(i+1)
        img_file = os.path.join(data_config['root_dir'], img_name)
        image = Image.open(img_file)
        image = image.convert("RGB")
        image_list.append(image)
    return image_list

def loadSampleImage(data_config):
    image_root = data_config['image_root']
    img = Image.open(image_root)
    img_array = np.array(img)
    return img_array


def loadData(data_type):
    # 하드코딩된 절대 경로 대신 os.path.join과 DATA_ROOT 사용
    dataset_configs = {
        'Kodak_images':{
            'load_fcn':loadDatasetImages,
            'num_images':24,
            'root_dir': os.path.join(DATA_ROOT, 'Kodak'),
            'fname_format':'kodim{:02}.png'
        },
        'seolyoon':{
            'load_fcn':loadSampleImage,
            'image_root':'images/samples/seolyoon_kam.png',
        },
        'seolyoonL':{
            'load_fcn':loadSampleImage,
            'image_root':'images/samples/seolyoon_large.png',
        },
        'cifar_sample':{
            'load_fcn':loadSampleImage,
            'image_root':'images/samples/cifar10.png',
        },
        'ImageNet_8000':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCodeCachedImageDataset,
            'transform_type':'train',
            'patch_size':(256, 256),
            'num_images':8000,
            'root_dir': os.path.join(DATA_ROOT, 'ImageNet8000'),
            'fname_format':'img_{}.jpg',
        },
        'ImageNet_8000_224':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCodeCachedImageDataset,
            'transform_type':'train',
            'patch_size':(224, 224),
            'num_images':8000,
            'root_dir': os.path.join(DATA_ROOT, 'ImageNet8000'),
            'fname_format':'img_{}.jpg',
        },
        'ImageNet_mini':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCachedImageDataset,
            'transform_type':'train',
            'patch_size':(256, 256),
            'num_images':100,
            'root_dir': os.path.join(DATA_ROOT, 'ImageNet8000'),
            'fname_format':'img_{}.jpg'
        },
        'Kodak':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCachedImageDataset,
            'transform_type':'test',
            'patch_size':(256, 256),
            'num_images':24,
            'root_dir': os.path.join(DATA_ROOT, 'Kodak'),
            'fname_format':'kodim{:02}.png'
        },
        'Kodak_rep':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCachedRepeatedImageDataset,
            'transform_type':'test',
            'patch_size':(256, 256),
            'num_images_per_ds':24,
            'num_reps':20,
            'root_dir': os.path.join(DATA_ROOT, 'Kodak'),
            'fname_format':'kodim{:02}.png'
        },
        'Kodak_valid':{
            'load_fcn':loadTorchDatasets,
            'dataset_class':indexedCachedImageDataset,
            'transform_type':'valid',
            'patch_size':(256, 256),
            'num_images':24,
            'root_dir': os.path.join(DATA_ROOT, 'Kodak'),
            'fname_format':'kodim{:02}.png'
        },
    }
    
    try:
        data_config = dataset_configs[data_type].copy()
        load_fcn = data_config.pop('load_fcn')
        data = load_fcn(data_config)
    except KeyError:
        raise KeyError(f"Key '{data_type}' does not exist! Try {', '.join(dataset_configs.keys())}")
    
    return data

def load_dataloaders(train_dataset, valid_dataset, train_batch_size, valid_batch_size, num_workers):
    train_ds = loadData(train_dataset) # ds refers to dataset
    valid_ds = loadData(valid_dataset)
    
    train_dl = DataLoader( # dl refers to dataloader
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=3,
        pin_memory=False,
        persistent_workers=True
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    
    return train_dl, valid_dl
    
