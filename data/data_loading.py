from datasets import load_dataset

def get_kvasir_data():
    print("--- Veri Seti İndiriliyor ---")
    dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA")
    if 'train' in dataset:
        return dataset['train']
    return dataset['raw']

def get_train_val_split(dataset, split_ratio=0.8):
    split_data = dataset.train_test_split(test_size=(1 - split_ratio))
    return split_data['train'], split_data['test']