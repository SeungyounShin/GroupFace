from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os,cv2,random
from itertools import combinations
import mxnet as mx
from tqdm import tqdm

class CustomFace(Dataset):
    def __init__(self,root,transform=None, align_path="/home/yo0n/workspace2/celebrity_lmk", sample=None):
        self.root = root
        self.identities = [i for i in os.listdir(self.root) if "DS" not in i]
        if sample is not None:
            self.identities = random.sample(self.identities, sample)
        self.img_paths = list()
        for iden in self.identities:
            self.img_paths += [root + "/"+ iden+ "/"+i for i in os.listdir(root + "/" + iden) if "DS" not in i]
        self.transform = transform
        self.align_path = align_path
        if self.align_path is not None:
            f = open(align_path, 'r')
            label_txt = f.read()
            f.close()
            self.label_txt = label_txt.split('\n')
            self.label_txt = [i.split(' ') for i in self.label_txt]
            self.label_dict = dict()
            self.pad = 112
            for i in tqdm(range(len(self.label_txt))):
                align_info = self.label_txt[i][2:]
                align_info = [float(j) for j in align_info]
                xalign = [j for idx,j in enumerate(align_info) if idx%2==0]
                yalign = [j for idx,j in enumerate(align_info) if (idx+1)%2==0]
                xs,ys = sum(xalign)/5.-self.pad, sum(yalign)/5.-self.pad
                self.label_dict[self.label_txt[i][0]] =  (int(xs),int(ys))
                #print(self.label_dict)


    def find_alignment(self,idx):
        img_path = self.img_paths[idx]
        id = img_path.split('/')[-2]
        imgP = img_path.split('/')[-1]
        query = id + "/" + imgP
        for i in range(len(self.label_txt)):
            if query in self.label_txt[i]:
                align_info = self.label_txt[idx].split(' ')[2:]
                align_info = [float(i) for i in align_info]
                xalign = [i for idx,i in enumerate(align_info) if idx%2==0]
                yalign = [i for idx,i in enumerate(align_info) if (idx+1)%2==0]
                xs,ys = sum(xalign)/5.-self.pad, sum(yalign)/5.-self.pad
                return int(xs),int(ys)
        return None

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        #print(img_path)
        iden = self.identities.index(img_path.split("/")[-2])
        img = cv2.imread(img_path)
        if self.align_path is not None:
            ys,xs = self.label_dict[img_path[img_path.index('celebrity'):]]
            img = img[xs:xs+self.pad*2, ys:ys+self.pad*2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)
            return img['image'],iden
        return img, iden


    def __len__(self):
        return len(self.img_paths)


class CustomFaceValid(Dataset):
    def __init__(self,root, transform=None):
        self.root = root
        self.identities = [i for i in os.listdir(self.root) if "DS" not in i]
        self.img_paths = list()
        for iden in self.identities:
            self.img_paths += [root + "/"+ iden+ "/"+i for i in os.listdir(root + "/" + iden) if "DS" not in i]
        self.transform = transform
        self.pairs = list(combinations(self.img_paths, 2))
        pos_pairs = [i for i in self.pairs if i[0].split('/')[-2] == i[1].split('/')[-2] ]
        neg_pairs = [i for i in self.pairs if i[0].split('/')[-2] != i[1].split('/')[-2] ]

        neg_pairs = random.sample(neg_pairs, k=len(pos_pairs))
        self.pairs = pos_pairs + neg_pairs


    def __getitem__(self, idx):
        pair_paths = self.pairs[idx]
        iden_path1, iden_path2 = pair_paths[0], pair_paths[1]
        iden1, iden2 = iden_path1.split('/')[-2], iden_path2.split('/')[-2]
        label = 0
        if iden1 == iden2:
            label = 1

        img1 = cv2.imread(iden_path1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, dsize=(112, 112), interpolation=cv2.INTER_AREA)

        img2 = cv2.imread(iden_path2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, dsize=(112, 112), interpolation=cv2.INTER_AREA)
        return img1, img2, label


    def __len__(self):
        return len(self.pairs)

class MS1M(Dataset):
    def __init__(self,root="/home/yo0n/workspace2/ms1m-retinaface-t1", transform=None):
        self.root = root
        self.record = mx.recordio.MXIndexedRecordIO("/home/yo0n/workspace2/ms1m-retinaface-t1/train.idx", "/home/yo0n/workspace2/ms1m-retinaface-t1/train.rec", 'r')
        #print(self.record.read_idx(20))
        self.transform = transform
        self.identities = [i for i in range(93431)]

    def __getitem__(self, idx):
        img_mxnet = self.record.read_idx(idx+1)
        header, img = mx.recordio.unpack_img(img_mxnet)

        iden = int(header.label[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)

        return img['image'], iden


    def __len__(self):
        return 5179509

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import random
    """
    ms1m = MS1M()
    idx = random.randint(0, len(ms1m))
    print(idx)
    img, iden = ms1m[0]

    print(iden)
    #plt.imshow(img)
    #plt.show()


    max = -9
    for i in range(5179509):
        img, iden = ms1m[i]
        if(max < iden):
            max = iden
            print(max)
    """
    dataset = CustomFace("/sdb/celebrity")
    while True:
        idx = random.randint(0, len(dataset))
        print(idx)
        img, iden = dataset[idx]

        print(iden)
        plt.imshow(img)
        plt.show()
