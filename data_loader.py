class HDF5Dataset(Dataset):

  def __init__(self, h5_path):
    self.path = h5_path
    self.data = test_catvnoncat.h5(self.path,'r')
    self.length = len(h5py.File(h5_path,'r'))
  
  def __getitem__(self, index):

    record = self.data[str(index)]
    image = record['X'].value

    img = Image.fromarray(pixels.astype('utf8'),'RGB')
    label = record['y'].value

    image = self.transform(img)
    return (img,label)
  
  def __len__(self):
    return self.length


train_loader = torch.utils.data.DataLoader('/content/aia_anger.h5', shuffle=True)
test_loader = torch.utils.data.DataLoader('/content/aia_joy.h5', shuffle=True)
