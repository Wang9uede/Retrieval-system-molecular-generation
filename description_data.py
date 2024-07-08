from molt5_dataset import Mol2CaptionDataset
import torch
from torch.utils.data import DataLoader
from molxpt_tokenizer import MolxptTokenizer

# 假设 raw_folder 和 pro_folder 是你的数据文件夹路径，mode 是你的数据模式
raw_folder = '/Users/wangjuede/Documents/mol2cap_data'
mode = 'train'  # 可以是 'train', 'val' 或 'test'

# 实例化数据集
dataset = Mol2CaptionDataset(raw_folder, mode)

# 创建 DataLoader
tokenizer = MolxptTokenizer.from_pretrained('molxpt_ckpt')
output = tokenizer.tokenize(dataset[1][1])
print(output)