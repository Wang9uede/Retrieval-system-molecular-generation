import torch
import re
from torch.utils.data import Dataset
from molxpt_tokenizer import MolxptTokenizer

class Mol2CaptionDataset(Dataset):
    def __init__(self, raw_folder, mode, tokenizer=None):  # 添加了 tokenizer 参数
        raw_file = raw_folder + '/{}.txt'.format(mode)
        with open(raw_file, 'r') as f:
            lines = f.readlines()

        lines = lines[1:]  # 假设第一行不是数据，跳过
        self.data = []
        for line in lines:
            temp = line.strip().split('\t')
            self.data.append([temp[-2], temp[-1]])  # 假设分子结构在倒数第二列，caption在最后一列

        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        molecule, caption = self.data[idx]
        if self.tokenizer:
            # 如果提供了 tokenizer，则使用它来编码 caption
            inputs = self.tokenizer(caption, return_tensors="pt").input_ids
            return molecule, inputs, caption
        else:
            # 如果没有提供 tokenizer，则只返回原始字符串
            return molecule, caption

       # return self.data[idx]
