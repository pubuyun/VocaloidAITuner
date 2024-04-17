import torch
import torch.nn as nn
from torch.nn import Transformer
import os
import pandas as pd
from torch.utils.data import Dataset
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ydict = ["a","ai","an","ang","ao","shi","liu","ge","ba","bai","ban","bang","bao","bei","ben","beng","bi","bian","biao","bie","bin","bing","po","bo","bu","san","si","ca","cai","can","cang","cao","ce","cen","ceng","cha","chai","chan","chang","zhang","chao","che","chen","cheng","chi","chong","chou","chu","chuai","chuan","chuang","chui","chun","chuo","ci","cong","cou","cu","cuan","cui","cun","cuo","er","da","dai","dan","dang","dao","de","deng","di","dian","diao","die","ding","diu","dong","dou","du","duan","dui","dun","duo","e","en","jiu","fa","fan","fang","fei","fen","feng","fu","fou","ga","gai","gan","gang","gao","gei","gen","geng","gong","gou","gu","gua","guai","guan","guang","gui","gun","guo","ha","hai","han","xing","hang","hao","he","hei","hen","heng","hong","hou","hu","hua","huai","huan","huang","hui","hun","huo","ji","jia","jian","jiang","jiao","jie","jin","jing","qing","jiong","ju","juan","jue","jun","ka","kai","kan","kang","kao","ke","ken","keng","kong","kou","ku","kua","kuai","kuan","kuang","kui","kun","kuo","wu","la","lai","lan","lang","lao","le","lei","leng","li","lia","lian","liang","liao","lie","lin","ling","long","lou","lu","lv","luan","lve","lun","luo","ma","mai","man","mang","mao","me","mei","men","meng","mi","mian","miao","mie","min","ming","mo","mou","mu","na","nai","nan","nang","nao","nei","nen","neng","ni","nian","niang","niao","nie","nin","ning","niu","nong","nu","nv","nuan","nve","nuo","yi","ou","qi","pa","pai","pan","pang","pao","pei","pen","peng","pi","pian","piao","pie","pin","ping","pou","pu","qia","qian","qiang","qiao","qin","qiong","qiu","qu","quan","que","qun","ran","rang","rao","re","ren","reng","ri","rong","rou","ru","ruan","rui","run","ruo","sa","sai","sang","sao","se","sen","seng","sha","shai","shan","shang","shao","she","shen","sheng","shou","shu","shua","shuai","shuan","shuang","shui","shun","shuo","song","sou","su","suan","sui","sun","suo","ta","tai","tan","tang","tao","te","teng","ti","tian","tiao","tie","ting","tong","tou","tu","tuan","tui","tun","wa","wai","wan","wang","wei","wen","weng","wo","xi","xia","xian","xiang","xiao","xie","xin","xiong","xiu","xu","xuan","xue","xun","ya","yan","yang","yao","ye","yin","ying","you","yong","yu","yuan","yue","yun","za","zai","zan","zang","zao","ze","zei","zen","zeng","zha","zhai","zhan","zhao","zhe","zhen","zheng","zhi","zhong","zhou","zhu","zhua","zhuai","zhuan","zhuang","zhui","zhun","zhuo","zi","zong","zou","zu","zuan","zui","zun","zuo","shei","chua","dei","den","ne","dia","fo","lo","miu","nou","o","qie","tuo","zhei","ei"]
class RNNDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        df = pd.read_csv(file_path)
        # get the inputs and targets
        inputs = df.iloc[:,3:]
        targets = df.iloc[:,:3]
        # covert to tensor
        inputs = torch.tensor(inputs.values, dtype=torch.float)
        targets = torch.tensor(targets.values, dtype=torch.float)
        return inputs, targets
train_dataset = RNNDataset('trainfiles')
# create a trainloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# define the model
# input_size : [n, 1 + len(ydict)]
# output_size : [n, 3]
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        # Packing
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM
        packed_output, _ = self.lstm(x_packed)
        
        # Unpacking
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Fully connected layer
        out = self.fc(output)
        return out

# Define your model
input_size = 1+len(ydict)  # Input size depends on the length of ydict
hidden_size = 50  # You can define it as per your requirement
num_layers = 2  # You can define it as per your requirement
output_size = 3  # Output size is 3 as per your requirement

model = RNN(input_size, hidden_size, num_layers, output_size)

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        # Get data
        sequences, labels = batch
        lengths = [len(seq) for seq in sequences]
        sequences = sequences.to(device)
        labels = labels.to(device)
        print(sequences.shape, len(lengths))
        # Forward pass
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss
# load the model
if os.path.exists("model.pth"):
    model = torch.load("model.pth")
    print("Model loaded")
# Define criterion and optimizer
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 512
# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
# save the model
torch.save(model, 'model.pth')
print("Model saved, path:model.pth")
# test the model
test_dataset = RNNDataset('testfiles')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
for batch in test_loader:
    sequences, labels = batch
    lengths = [len(seq) for seq in sequences]
    sequences = sequences.to(device)
    labels = labels.to(device)
    outputs = model(sequences, lengths)
    loss = criterion(outputs, labels)
    print("test loss: ", loss.item())