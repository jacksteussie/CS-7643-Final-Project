from torch.utils.data import Dataset, DataLoader

class XViewDataset(Dataset):
    def __init__(self, img_dir, json_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open()