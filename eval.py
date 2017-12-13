import torch
from torch.autograd import Variable

import dataset
from dataset import SifDataset


data = SifDataset()

checkpoint = torch.load('model.pth')
model = checkpoint['net'].eval()
dictionary = checkpoint['dict']
# print('Loaded model {}, reported w/ accuracy {}'.format(model, checkpoint['score']))
test_loss = 0
correct = 0
exs, labels = dataset.preprocess_dataset(data.dev, dictionary)
for data, target in zip(exs, labels):
    data, target = Variable(data.unsqueeze(0), volatile=True), Variable(target)
    output = model(data)
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()

test_loss /= len(exs)

print("%.1f" % (100 * correct / len(exs)))
