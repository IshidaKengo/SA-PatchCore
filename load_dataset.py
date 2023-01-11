from torch.utils.data import Dataset
import os
import glob
from PIL import Image


class load_Dataset(Dataset):
    def __init__(self, root, transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            #self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        #self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        #gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == '0_good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                #gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['0_good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                #gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                #gt_paths.sort()
                img_tot_paths.extend(img_paths)
                #gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        #assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        #if gt == 0:
        #    gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        #else:
        #    gt = Image.open(gt)
        #    gt = self.gt_transform(gt)
        
        #assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, label, os.path.basename(img_path[:-4]), img_type
