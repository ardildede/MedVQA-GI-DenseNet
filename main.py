%%writefile vqa_project/main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Importlar
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import DenseNet_BERT_VQA

# --- GRAFİK ÇİZME FONKSİYONU ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('DenseNet + BERT Confusion Matrix')
    
    plt.savefig("densenet_matrix.png", dpi=300)
    print("📊 Karmaşıklık matrisi 'densenet_matrix.png' olarak kaydedildi.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Doku Uzmanı (DenseNet) Başlatılıyor ---")
    print(f"Kullanılan Cihaz: {device}")

    # 1. HAZIRLIK
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data)
    
    print("Sınıflar analiz ediliyor...")
    answers_list = train_data['answer']
    all_answers = sorted(list(set(str(ans).lower() for ans in answers_list)))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}
    print(f"Tespit Edilen Sınıf Sayısı: {len(answer_map)}")

    # 2. DATASET
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = KvasirHFDataset(train_data, answer_map, transform=transform)
    val_ds = KvasirHFDataset(val_data, answer_map, transform=transform)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['answer'] for item in batch])
        raw_questions = [item['question'] for item in batch]
        
        tokenized = tokenizer(raw_questions, padding=True, truncation=True, 
                              max_length=25, return_tensors="pt")
        
        return images, tokenized['input_ids'], tokenized['attention_mask'], labels

    # Kaggle Ayarları
    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 3. MODEL KURULUMU
    model = DenseNet_BERT_VQA(num_classes=len(answer_map), freeze_bert=False).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # --- AKILLI KAYDETME SİSTEMİ ---
    MODEL_PATH = "densenet_model.pth"

    if os.path.exists(MODEL_PATH):
        print(f"\n✅ Kayıtlı model bulundu ({MODEL_PATH}). Eğitim ATLANIYOR...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except:
            model.load_state_dict(torch.load(MODEL_PATH), strict=False)
            print("⚠️ Model esnek modda yüklendi.")
    else:
        print("\n--- Eğitim Başlıyor ---")
        EPOCHS = 3
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for imgs, input_ids, mask, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)
                input_ids, mask = input_ids.to(device), mask.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs, input_ids, mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} Bitti. Loss: {total_loss/len(train_loader):.4f}")
        
        # Eğitimi Kaydet
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Model başarıyla kaydedildi: {MODEL_PATH}")

    # 4. TEST VE RAPORLAMA
    print("\n--- Test Raporu Hazırlanıyor ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, input_ids, mask, labels in tqdm(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            input_ids, mask = input_ids.to(device), mask.to(device)
            
            outputs = model(imgs, input_ids, mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Listeleri Hazırla
    label_ids = list(answer_map.values())
    label_names = list(answer_map.keys())

    # A) TXT RAPORU
    report_text = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0)
    with open("densenet_rapor.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("✅ 'densenet_rapor.txt' kaydedildi.")

    # B) EXCEL/CSV RAPORU
    report_dict = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0,
                                        output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv("densenet_analiz.csv")
    print("✅ 'densenet_analiz.csv' kaydedildi (Excel formatı).")
    
    # C) GRAFİK
    plot_confusion_matrix(all_labels, all_preds, label_names)
    
    print("\n🎉 Tüm işlemler tamamlandı. Output klasörünü kontrol et!")

if __name__ == "__main__":
    main()