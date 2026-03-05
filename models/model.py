%%writefile vqa_project/models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel

class DenseNet_BERT_VQA(nn.Module):
    def __init__(self, num_classes, freeze_bert=False):
        super(DenseNet_BERT_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: DenseNet121 (Doku detayları için)
        print("Model: DenseNet121 (ImageNet Pretrained) yükleniyor...")
        densenet = models.densenet121(pretrained=True)
        self.densenet_features = densenet.features
        self.img_fc = nn.Linear(1024, 512) # DenseNet çıktısı 1024 kanaldır
        
        # 2. METİN: DistilBERT
        print("Model: DistilBERT yükleniyor...")
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.text_fc = nn.Linear(768, 512)
        
        # 3. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Resim (DenseNet)
        features = self.densenet_features(images)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        img_emb = self.img_fc(out)
        
        # Metin (BERT)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]
        text_emb = self.text_fc(text_emb)
        
        # Birleştir
        combined = torch.cat((img_emb, text_emb), dim=1)
        output = self.classifier(combined)
        
        return output