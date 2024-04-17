import argparse
import vsqxt
import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def normalizer(dataArray):
    if dataArray.max() - dataArray.min() == 0:
        return dataArray
    return (dataArray-dataArray.min())/(dataArray.max() - dataArray.min())

ydict = ["a","ai","an","ang","ao","shi","liu","ge","ba","bai","ban","bang","bao","bei","ben","beng","bi","bian","biao","bie","bin","bing","po","bo","bu","san","si","ca","cai","can","cang","cao","ce","cen","ceng","cha","chai","chan","chang","zhang","chao","che","chen","cheng","chi","chong","chou","chu","chuai","chuan","chuang","chui","chun","chuo","ci","cong","cou","cu","cuan","cui","cun","cuo","er","da","dai","dan","dang","dao","de","deng","di","dian","diao","die","ding","diu","dong","dou","du","duan","dui","dun","duo","e","en","jiu","fa","fan","fang","fei","fen","feng","fu","fou","ga","gai","gan","gang","gao","gei","gen","geng","gong","gou","gu","gua","guai","guan","guang","gui","gun","guo","ha","hai","han","xing","hang","hao","he","hei","hen","heng","hong","hou","hu","hua","huai","huan","huang","hui","hun","huo","ji","jia","jian","jiang","jiao","jie","jin","jing","qing","jiong","ju","juan","jue","jun","ka","kai","kan","kang","kao","ke","ken","keng","kong","kou","ku","kua","kuai","kuan","kuang","kui","kun","kuo","wu","la","lai","lan","lang","lao","le","lei","leng","li","lia","lian","liang","liao","lie","lin","ling","long","lou","lu","lv","luan","lve","lun","luo","ma","mai","man","mang","mao","me","mei","men","meng","mi","mian","miao","mie","min","ming","mo","mou","mu","na","nai","nan","nang","nao","nei","nen","neng","ni","nian","niang","niao","nie","nin","ning","niu","nong","nu","nv","nuan","nve","nuo","yi","ou","qi","pa","pai","pan","pang","pao","pei","pen","peng","pi","pian","piao","pie","pin","ping","pou","pu","qia","qian","qiang","qiao","qin","qiong","qiu","qu","quan","que","qun","ran","rang","rao","re","ren","reng","ri","rong","rou","ru","ruan","rui","run","ruo","sa","sai","sang","sao","se","sen","seng","sha","shai","shan","shang","shao","she","shen","sheng","shou","shu","shua","shuai","shuan","shuang","shui","shun","shuo","song","sou","su","suan","sui","sun","suo","ta","tai","tan","tang","tao","te","teng","ti","tian","tiao","tie","ting","tong","tou","tu","tuan","tui","tun","wa","wai","wan","wang","wei","wen","weng","wo","xi","xia","xian","xiang","xiao","xie","xin","xiong","xiu","xu","xuan","xue","xun","ya","yan","yang","yao","ye","yin","ying","you","yong","yu","yuan","yue","yun","za","zai","zan","zang","zao","ze","zei","zen","zeng","zha","zhai","zhan","zhao","zhe","zhen","zheng","zhi","zhong","zhou","zhu","zhua","zhuai","zhuan","zhuang","zhui","zhun","zhuo","zi","zong","zou","zu","zuan","zui","zun","zuo","shei","chua","dei","den","ne","dia","fo","lo","miu","nou","o","qie","tuo","zhei","ei"]
def preprocess(filename):
    data = {"Pitchs":[], "Lyrics":[]}
    try:
        vtracks = vsqxt.vsqx.read(filename)
    except IndexError:
        print('Wrong version')
    for track in vtracks.vsTrack:
        # process note-related data
        length = int(track.return_all_note()[-1].t) + int(track.return_all_note()[-1].dur)

        N = np.array([None] * (length+1))
        Y = [None] * (length+1)
        for note in track.return_all_note():
            N[int(note.t):int(note.t)+int(note.dur)] = [int(note.n)] * int(note.dur)
            # using one-hot encoding for lyrics
            if note.y in ydict:
                Y[int(note.t):int(note.t)+int(note.dur)] = [[1 if ydict.index(note.y) == i else 0 for i in range(len(ydict))] ] * int(note.dur) 
            else:
                Y[int(note.t):int(note.t)+int(note.dur)] = [[0] * len(ydict)] * int(note.dur)
        data['Pitchs'].append(N)
        data["Lyrics"].append({})
        # add lyrics to the data
        for i in range(len(ydict)):
            data["Lyrics"][-1][ydict[i]] = [y[i] if y is not None else 0 for y in Y ]
    return data, vtracks

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

def predict(data):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    model = torch.load('model.pth')
    model = model.to(device)
    output = []
    # predict
    for N, Y in zip(data['Pitchs'], data['Lyrics']):
        with torch.no_grad():
            N = torch.tensor([n if n is not None else 0 for n in N])
            N = torch.unsqueeze(N, 0)
            Y = torch.tensor([y for y in Y.values()])
            sequences = torch.cat((N, Y), 0)
            sequences = torch.unsqueeze(sequences, 0).to(device)
            sequences = sequences.transpose(1,2).float()
            out = torch.tensor([]).to(device)
            # torch.Size([1, 146161]) torch.Size([406, 146161])
            # torch.Size([1, 407, 146161])
            for batch in torch.chunk(sequences, chunks=10, dim=1):
                lengths = [len(seq) for seq in batch]
                o = model(batch, lengths)
                out = torch.cat((out, o), 1)
            out = out.squeeze(dim=0)
            out = out.transpose(0,1)
            output.append(out)
    return output

def postprocess(output, vtracks:vsqxt.vsqx.vsTrack):
    for i, track in enumerate(vtracks.vsTrack):
        D = normalizer(np.array(output[i][0].cpu()))*128
        P = normalizer(np.array(output[i][1].cpu()))*8192
        V = normalizer(np.array(output[i][2].cpu()))*128
        print(track.vsPart[0].t, track.vsPart[0].playTime)  
        for i, d in enumerate(D):
            try:    
                d = int(d)
                if d != int(D[i-1]):
                    track.create_cc('D', d, i)
            except vsqxt.base.myError:
                pass
        for i, p in enumerate(P):
            try:
                p = int(p)
                if p != int(P[i-1]):
                    track.create_cc('P', p, i)
            except vsqxt.base.myError:
                pass

        for note in track.return_all_note():
            note.v = int(V[int(note.t)])
        print(len(track.return_all_cc()))
    return vtracks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default = None)
    parser.add_argument('--o', type=str, default = 'result.vsqx')
    args = parser.parse_args()
    data, vtracks = preprocess(args.i)
    output = predict(data)
    vtracks = postprocess(output, vtracks)
    vtracks.write(args.o)
