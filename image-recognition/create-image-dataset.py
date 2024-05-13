from datasets import load_dataset

dataset = load_dataset('imagefolder', data_dir='../image-downloader')

dataset.save_to_disk('./build/tarkov-items-image-dataset')