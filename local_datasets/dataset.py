import torch
from torch.utils.data import Dataset

class KvasirHFDataset(Dataset):
    def __init__(self, hf_dataset, answer_map, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.answer_map = answer_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        # DenseNet ve diğer CNN modelleri için transform (Resize, Normalize vb.) burada uygulanır
        if self.transform:
            image = self.transform(image)
            
        question = item['question']
        answer = str(item['answer']).lower()
        label = self.answer_map.get(answer, 0)

        return {
            'image': image, 
            'question': question, # Ham metin (BERT Tokenizer main.py'de işleyecek)
            'answer': torch.tensor(label, dtype=torch.long)
        }