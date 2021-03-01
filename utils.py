import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
import scipy.io
import glob


class Dogs(data.Dataset):
    """Get dataset for PyTorch"""

    def __init__(self, train=True, input_size=256):
        """Initialisation"""
        dataset_dir = '../Datasets/StanfordDogs/'
        self.image_dir = os.path.join(dataset_dir, 'Images-cropped')
        list_dir = os.path.join(dataset_dir, 'lists')
        if train:
            self.list_IDs = self._list_ids(list_dir, 'train_list.mat')
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.RandomHorizontalFlip(),
                                                  # transforms.ToTensor(),
                                                  # transforms.Resize(input_size),
                                                  # transforms.RandomCrop(input_size),
                                                  transforms.RandomResizedCrop(input_size, scale=(0.85, 1.15),
                                                                               ratio=(1., 1.)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])
        else:
            # self.list_IDs = self._list_ids(list_dir, 'test_list.mat')
            self.list_IDs = self._list_ids(list_dir, 'train_list.mat')
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.ToTensor(),
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])

    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample from dataset"""
        image_path = self.list_IDs[index]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        image = self.transforms(image)
        return image, image_path

    def _list_ids(self, list_dir, list_file):
        scipy_mat = scipy.io.loadmat(os.path.join(list_dir, list_file))['file_list']
        list_ids = []
        for file in scipy_mat:
            file_path = file[0][0]
            if os.path.exists(os.path.join(self.image_dir, file_path)):
                list_ids.append(file_path)
        return list_ids


class Cats(data.Dataset):
    """Get dataset for PyTorch"""

    def __init__(self, train=True, input_size=256):
        """Initialisation"""
        dataset_dir = '../Datasets/Cats/'
        self.image_dir = os.path.join(dataset_dir, 'Images-cropped')
        if train:
            self.list_IDs = self._read_txt_file(dataset_dir, 'train.txt')
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.RandomHorizontalFlip(),
                                                  # transforms.ToTensor(),
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])
        else:
            # self.list_IDs = self._list_ids(list_dir, 'test_list.mat')
            self.list_IDs = self._read_txt_file(dataset_dir, 'train.txt')
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.ToTensor(),
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])

    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample from dataset"""
        image_path = self.list_IDs[index]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        image = self.transforms(image)
        return image, image_path

    @staticmethod
    def _read_txt_file(data_dir, txt_file):
        list_ids = []
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                if os.path.exists(line[0]):
                    path = line[0].split(os.sep)
                    list_ids.append(os.path.join(path[-2], path[-1]))
        return list_ids


class Disney(data.Dataset):
    """Get dataset for PyTorch"""

    def __init__(self, train=True, input_size=256):
        """Initialisation"""
        dataset_dir = '../Datasets/Disney/disney-princess-colour-line-silhouette/Ariel/Colour'

        print("Currently Disney Dataset only Implemented for Ariel")
        # select a princess with most examples....

        # Get all ids
        list_ids = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))

        # Randomly partition to train (80%) / test (20)
        # save txts
        split_id = int(0.8*len(list_ids))
        train_ids = list_ids[:split_id]
        test_ids = list_ids[split_id:]
        # self.save_txt(train_ids, 'ariel_train.txt')
        # self.save_txt(test_ids, 'ariel_test.txt')

        self.image_dir = dataset_dir
        if train:
            self.list_IDs = train_ids
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.RandomHorizontalFlip(),
                                                  # transforms.ToTensor(),
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])
        else:
            self.list_IDS = test_ids
            self.transforms = transforms.Compose([
                                                  # transforms.Resize((input_size, input_size)),
                                                  # transforms.ToTensor(),
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])

    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample from dataset"""
        image_path = self.list_IDs[index]
        image = Image.open(os.path.join(image_path)).convert('RGB')
        image = self.transforms(image)
        return image, image_path

    # @staticmethod
    # def _read_txt_file(data_dir, txt_file):
    #     list_ids = []
    #     with open(os.path.join(data_dir, txt_file), 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.split(' ')
    #             if os.path.exists(line[0]):
    #                 path = line[0].split(os.sep)
    #                 list_ids.append(os.path.join(path[-2], path[-1]))
    #     return list_ids

    def save_txt(self, list_ids, outpath):
        pass


class Celeba(data.Dataset):
    """Get dataset for PyTorch"""

    def __init__(self, train=True, input_size=256):
        """Initialisation"""
        dataset_dir = '../Datasets/CelebA/'
        self.image_dir = os.path.join(dataset_dir, 'Images-cropped', 'img_celeba')
        if train:
            self.list_IDs = self._read_txt_file(dataset_dir, 'train.txt')
            # Resize smallest size to 256 then crop centre.
            self.transforms = transforms.Compose([
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])
        else:
            # self.list_IDs = self._list_ids(list_dir, 'test_list.mat')
            self.list_IDs = self._read_txt_file(dataset_dir, 'train.txt')
            self.transforms = transforms.Compose([
                                                  transforms.Resize(input_size),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.ToTensor(),
                                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  #                      std=[0.229, 0.224, 0.225])
                                                  ])

    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample from dataset"""
        image_path = self.list_IDs[index]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        image = self.transforms(image)
        return image, image_path

    @staticmethod
    def _read_txt_file(data_dir, txt_file):
        list_ids = []
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                if os.path.exists(line[0]):
                    path = line[0].split(os.sep)
                    list_ids.append(os.path.join(path[-2], path[-1]))
        return list_ids


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                # if p.grad.abs().sum() == 0.:
                print(n, p.grad.abs().mean())

    # print(ave_grads.mean())
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)