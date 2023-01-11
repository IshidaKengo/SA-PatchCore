import argparse
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
import csv

from load_dataset import load_Dataset
from utils import *

        
class MODEL(pl.LightningModule):
    def __init__(self, hparams):
        super(MODEL, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        #print(self.model)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()
        
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []      
        self.anomaly_map_all = []    
        self.input_x_list =[]
        self.defect_types = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features
    
    def save_anomaly_map(self, anomaly_map, input_img, name, saturate_hi_num, saturate_lw_num, save_dir):
        
        for i in range(len(input_img)):
            if anomaly_map[i].shape != input_img[i].shape:
                anomaly_map[i] = cv2.resize(anomaly_map[i], (input_img[i].shape[0], input_img[i].shape[1]))
            anomaly_map_norm = select_min_max_norm(anomaly_map[i], saturate_hi_num, saturate_lw_num)
            anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

            # anomaly map on image
            hm_on_img = heatmap_on_image(anomaly_map_norm_hm , input_img[i])
            
            # save images
            cv2.imwrite(os.path.join(save_dir, f'{name}_{str(i)}.jpg'), input_img[i])
            cv2.imwrite(os.path.join(save_dir, f'{name}_{str(i)}_amap.jpg'), anomaly_map_norm_hm)
            cv2.imwrite(os.path.join(save_dir, f'{name}_{str(i)}_amap_on_img.jpg'), hm_on_img)
            #cv2.imwrite(os.path.join(self.result_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)


    def train_dataloader(self):
        image_datasets = load_Dataset(root=args.dataset_path, transform=self.data_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=1) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = load_Dataset(root=args.dataset_path, transform=self.data_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.result_path= prep_dirs(args.output_path)
        self.embedding_list = []
    
    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.result_path= prep_dirs(args.output_path)
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            #print(feature.shape)
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)
            
            if feature.shape[3] == 28:
                saconv_out = avep(feature) 

            elif feature.shape[3] ==14:
                attention_in1 = maxp(feature[:])
                attention_in2 = maxp(feature[:])
                attention_v = maxp(feature[:]) 

                width = attention_in1.shape[3]
        
                attention_in1_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_in2_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_v = torch.flatten(attention_v[:], 2, 3)
                attention_in1 = torch.reshape(attention_in1_flatten, (attention_in1_flatten.shape[0], attention_in1_flatten.shape[1], attention_in1_flatten.shape[2], 1))
                attention_in2 = torch.reshape(attention_in2_flatten, (attention_in2_flatten.shape[0], attention_in2_flatten.shape[1], 1, attention_in2_flatten.shape[2]))
                attention_v = torch.reshape(attention_v, (attention_v.shape[0], attention_v.shape[1], attention_v.shape[2], 1))
                attention_out = torch.matmul(attention_in1, attention_in2)
                attention_out = torch.nn.functional.softmax(attention_out, dim=-1)
                attention_out = torch.matmul(attention_out, attention_v)
                saconv_out = torch.reshape(attention_out, (attention_out.shape[0], attention_out.shape[1], width, width)) 
            
            embeddings.append(saconv_out)
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        print(total_embeddings.shape)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma  
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        print(total_embeddings.shape)
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        #print('initial embedding size : ', total_embeddings.shape)
        #print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        x, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)

            if feature.shape[3] == 28:
                saconv_out = avep(feature) 

            elif feature.shape[3] == 14:
                attention_in1 = maxp(feature[:])
                attention_in2 = maxp(feature[:])
                attention_v = maxp(feature[:]) 

                width = attention_in1.shape[3]
                
                attention_in1_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_in2_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_v = torch.flatten(attention_v[:], 2, 3)
                attention_in1 = torch.reshape(attention_in1_flatten, (attention_in1_flatten.shape[0], attention_in1_flatten.shape[1], attention_in1_flatten.shape[2], 1))
                attention_in2 = torch.reshape(attention_in2_flatten, (attention_in2_flatten.shape[0], attention_in2_flatten.shape[1], 1, attention_in2_flatten.shape[2]))
                attention_v = torch.reshape(attention_v, (attention_v.shape[0], attention_v.shape[1], attention_v.shape[2], 1))
                attention_out = torch.matmul(attention_in1, attention_in2)
                attention_out = torch.nn.functional.softmax(attention_out, dim=-1)
                attention_out = torch.matmul(attention_out, attention_v)
                saconv_out = torch.reshape(attention_out, (attention_out.shape[0], attention_out.shape[1], width, width))

            embeddings.append(saconv_out)
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=9)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        anomaly_map = score_patches[:,0].reshape((28,28))  

        score = max(score_patches[:,0]) # Image-level score
                
        #gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.anomaly_map_all.append(anomaly_map_resized_blur)
        
        #self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])

        self.pred_list_img_lvl.append(score) 

        self.img_path_list.extend(file_name)
        self.defect_types.append(x_type[0])
       
        x = self.inv_normalize(x)
        x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        x = np.uint8(min_max_norm(x)*255)
        self.input_x_list.append(x)        

    def test_epoch_end(self, outputs):
        
        a = self.defect_types.count('0_good') #210
        b = self.defect_types.count('1_scratch') #41
        c = self.defect_types.count('2_paint') #41
        d = self.defect_types.count('3_over-coupling') #44
        e = self.defect_types.count('4_lacking') #40
        
        num_defect_types = [a,b,c,d,e]
        name_defect_types = ['good', 'scratch', 'paint', 'over-coupling', 'lacking']

        ## caliculate AUC
        
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        
        #values = {'img_auc': img_auc}
        #self.log_dict(values)

        with open(self.result_path + r'/AUROC.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['All', img_auc])
        
        front, rear = a,a 
        for (length, name) in zip(num_defect_types[1:], name_defect_types[1:]):
            rear += length

            labels= np.concatenate([self.gt_list_img_lvl[0:a],
                                        self.gt_list_img_lvl[front:rear]], axis = 0)
            scores = np.concatenate([self.pred_list_img_lvl[0:a],
                                        self.pred_list_img_lvl[front:rear]], axis = 0)
            img_auc = roc_auc_score(labels, scores)
            #values = {'img_auc': img_auc}
            #self.log_dict(values)

            with open(self.result_path + r'/AUROC.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([name, img_auc])
        
            front += length

        ## SAVE ANOMALY MAPS
        if args.save_anomaly_map:
            temp_map = np.array(self.anomaly_map_all)
            max_num = temp_map.max()
            min_num = temp_map.min()
                
            saturate_hi_num = (max_num - min_num) * 1.0 #upper limit of anomaly maps
            saturate_lw_num = (max_num - min_num) * 0.7 #lower limit of anomaly maps
            
            save_dir = os.path.join(self.result_path, 'anomaly maps')
            if os.path.isdir(save_dir) == False:
                os.mkdir(save_dir)
            
            self.save_anomaly_map(self.anomaly_map_all[0:a], self.input_x_list[0:a], 'good', saturate_hi_num, saturate_lw_num, save_dir)
            self.save_anomaly_map(self.anomaly_map_all[a:a+b], self.input_x_list[a:a+b], 'scratch', saturate_hi_num, saturate_lw_num, save_dir)
            self.save_anomaly_map(self.anomaly_map_all[a+b:a+b+c], self.input_x_list[a+b:a+b+c], 'paint', saturate_hi_num, saturate_lw_num, save_dir)
            self.save_anomaly_map(self.anomaly_map_all[a+b+c:a+b+c+d], self.input_x_list[a+b+c:a+b+c+d], 'over-coupling', saturate_hi_num, saturate_lw_num, save_dir)
            self.save_anomaly_map(self.anomaly_map_all[a+b+c+d:a+b+c+d+e], self.input_x_list[a+b+c+d:a+b+c+d+e], 'lacking', saturate_hi_num, saturate_lw_num, save_dir)

                
def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./dataset/Co-occurrence Anomaly Detection Screw Dataset') 
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--load_size', default=256) 
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--output_path', default=r'./outputs') 
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.output_path, max_epochs=args.num_epochs, gpus = (device=="cuda") )
    model = MODEL(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)



        
        
