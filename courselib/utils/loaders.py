import os
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader

def load_or_download_csv(file_name, url, column_names=None, encoding='utf-8'):
    if os.path.exists(file_name):
        print(f"Loading from local `{file_name}`...")
        return pd.read_csv(file_name, index_col=0, encoding=encoding)
    else:
        print(f"Downloading from `{url}`...")
        df = pd.read_csv(url, names=column_names, encoding=encoding)
        df.to_csv(file_name, encoding=encoding)
        print("Saved to local file.")
        return df
    
def load_csv(file_name, encoding='utf-8'):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File `{file_name}` not found. Please make sure the file exists in the current working directory.")

    print(f"Loading from `{file_name}`...")
    return pd.read_csv(file_name, index_col=0, encoding=encoding)

def load_irises():
    file_name = 'iris_data_set.csv'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    column_names = ['sepal length [cm]', 'sepal width [cm]',
                'petal length [cm]', 'petal width [cm]', 'iris type']
    
    df = load_or_download_csv(file_name, url, column_names)

    return df

def load_digits():
    file_name = 'letter-recognition.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
    column_names = ['letter','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar',
                'x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']


    df = load_or_download_csv(file_name, url, column_names)
    return df 

def load_wine():
    file_name = 'wine.csv'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    column_names = ['Class','Alcohol', 'Malic acid','Ash', 'Alcalinity of ash', 'Magnesium',
               'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    df = load_or_download_csv(file_name, url, column_names)
    return df 

def load_music_30_sec():
    file_name = 'data/features_30_sec.csv'
    df = load_csv(file_name)
    return df 

def load_music_3_sec():
    file_name = 'data/features_3_sec.csv'
    df = load_csv(file_name)
    return df

def load_spectrograms(config_cnn):
    transform_train = transforms.Compose([
        transforms.Resize((config_cnn["img_size"], config_cnn["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config_cnn["normalize_mean"], std=config_cnn["normalize_std"])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((config_cnn["img_size"], config_cnn["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config_cnn["normalize_mean"], std=config_cnn["normalize_std"])
    ])
    dataset = datasets.ImageFolder(root=config_cnn["data_path"], transform=transform_val)

    return dataset, transform_train

def get_data_loaders(train_dataset, val_dataset, batch_size, shuffle=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader