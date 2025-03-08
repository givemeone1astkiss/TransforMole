from torch.utils.data import DataLoader
from moses import CharVocab, StringDataset, get_dataset

train = get_dataset('train')
vocab = CharVocab.from_data(train)
train_dataset = StringDataset(vocab, train)
train_dataloader = DataLoader(
    train_dataset, batch_size=512,
    shuffle=True, collate_fn=train_dataset.default_collate
)
print(vocab[0])