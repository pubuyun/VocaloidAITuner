import vsqxt
import numpy as np
import pandas as pd
import os

# define a normaliser
def normalizer(dataArray):
    if dataArray.max() - dataArray.min() == 0:
        return dataArray
    return (dataArray-dataArray.min())/(dataArray.max() - dataArray.min())

# define a standardizer
def standardizer(dataArray):
    if dataArray.max() - dataArray.min() == 0:
        return dataArray
    return (dataArray-dataArray.mean())/dataArray.std()

# rename the files in ./vcfiles in to n.vsqx
for root, dirs, files in os.walk('./vcfiles'):
    num = len(files)
    for i, file in enumerate(files, 1):
        os.rename(os.path.join(root, file), os.path.join(root, str(i)+'.vsqx'))

# 
ydict = ["a","ai","an","ang","ao","shi","liu","ge","ba","bai","ban","bang","bao","bei","ben","beng","bi","bian","biao","bie","bin","bing","po","bo","bu","san","si","ca","cai","can","cang","cao","ce","cen","ceng","cha","chai","chan","chang","zhang","chao","che","chen","cheng","chi","chong","chou","chu","chuai","chuan","chuang","chui","chun","chuo","ci","cong","cou","cu","cuan","cui","cun","cuo","er","da","dai","dan","dang","dao","de","deng","di","dian","diao","die","ding","diu","dong","dou","du","duan","dui","dun","duo","e","en","jiu","fa","fan","fang","fei","fen","feng","fu","fou","ga","gai","gan","gang","gao","gei","gen","geng","gong","gou","gu","gua","guai","guan","guang","gui","gun","guo","ha","hai","han","xing","hang","hao","he","hei","hen","heng","hong","hou","hu","hua","huai","huan","huang","hui","hun","huo","ji","jia","jian","jiang","jiao","jie","jin","jing","qing","jiong","ju","juan","jue","jun","ka","kai","kan","kang","kao","ke","ken","keng","kong","kou","ku","kua","kuai","kuan","kuang","kui","kun","kuo","wu","la","lai","lan","lang","lao","le","lei","leng","li","lia","lian","liang","liao","lie","lin","ling","long","lou","lu","lv","luan","lve","lun","luo","ma","mai","man","mang","mao","me","mei","men","meng","mi","mian","miao","mie","min","ming","mo","mou","mu","na","nai","nan","nang","nao","nei","nen","neng","ni","nian","niang","niao","nie","nin","ning","niu","nong","nu","nv","nuan","nve","nuo","yi","ou","qi","pa","pai","pan","pang","pao","pei","pen","peng","pi","pian","piao","pie","pin","ping","pou","pu","qia","qian","qiang","qiao","qin","qiong","qiu","qu","quan","que","qun","ran","rang","rao","re","ren","reng","ri","rong","rou","ru","ruan","rui","run","ruo","sa","sai","sang","sao","se","sen","seng","sha","shai","shan","shang","shao","she","shen","sheng","shou","shu","shua","shuai","shuan","shuang","shui","shun","shuo","song","sou","su","suan","sui","sun","suo","ta","tai","tan","tang","tao","te","teng","ti","tian","tiao","tie","ting","tong","tou","tu","tuan","tui","tun","wa","wai","wan","wang","wei","wen","weng","wo","xi","xia","xian","xiang","xiao","xie","xin","xiong","xiu","xu","xuan","xue","xun","ya","yan","yang","yao","ye","yin","ying","you","yong","yu","yuan","yue","yun","za","zai","zan","zang","zao","ze","zei","zen","zeng","zha","zhai","zhan","zhao","zhe","zhen","zheng","zhi","zhong","zhou","zhu","zhua","zhuai","zhuan","zhuang","zhui","zhun","zhuo","zi","zong","zou","zu","zuan","zui","zun","zuo","shei","chua","dei","den","ne","dia","fo","lo","miu","nou","o","qie","tuo","zhei","ei"]
n = 0
for j in range(1, num+1):
    try:
        vtracks = vsqxt.vsqx.read('vcfiles/'+str(j)+'.vsqx')
    except IndexError:
        print('Wrong version')
        continue
    # print progress
    print(j*100/num, '%')
    try:
        for track in vtracks.vsTrack:
            # process time-related data
            length = max(int(track.return_all_cc()[-1].t), int(track.return_all_note()[-1].t) + int(track.return_all_note()[-1].dur))
            print(length)
            D = np.array([None] * (length+1))
            P = np.array([None] * (length+1))
            S = np.array([None] * (length+1))
            for vcc in track.return_all_cc():
                if vcc.ID == 'D':
                    D[int(vcc.t)] = int(vcc.v)
                elif vcc.ID == 'P':
                    P[int(vcc.t)] = int(vcc.v)
                elif vcc.ID == 'S':
                    S[int(vcc.t)] = int(vcc.v)
            # Fill the gaps between changes             
            for i in range(length+1):
                if D[i] is None and i > 0:
                    D[i] = D[i-1]
                if P[i] is None and i > 0:
                    P[i] = P[i-1]
                if S[i] is None and i > 0:
                    S[i] = S[i-1]
            # replace None with 2 in S
            S = [2 if s is None else s for s in S]
            # replace None with 0 in P
            P = [0 if p is None else p for p in P]
            # replace None with 64 in D
            D = [63 if d is None else d for d in D]
            D = np.array(D)
            P = np.array(P)
            S = np.array(S)
            realP = P * S
            
            # process note-related data
            V = np.array([None] * (length+1))
            N = np.array([None] * (length+1))
            Y = [None] * (length+1)
            for note in track.return_all_note():
                V[int(note.t):int(note.t)+int(note.dur)] = [int(note.v)] * int(note.dur)
                N[int(note.t):int(note.t)+int(note.dur)] = [int(note.n)] * int(note.dur)
                # using one-hot encoding for lyrics
                if note.y in ydict:
                    Y[int(note.t):int(note.t)+int(note.dur)] = [[1 if ydict.index(note.y) == i else 0 for i in range(len(ydict))] ] * int(note.dur) 
                else:
                    Y[int(note.t):int(note.t)+int(note.dur)] = [[0] * len(ydict)] * int(note.dur)
            # replace None with 64 in V
            V = [64 if v is None else v for v in V]
            V = np.array(V)
            # replace None with 0 in N
            N = [0 if n is None else n for n in N]
            N = np.array(N)

            # delete D[i] and realP[i] if N[i] is None
            i = 0
            while None in Y:
                if Y[i] is None:
                    D = np.delete(D, i, axis=0)
                    realP = np.delete(realP, i)
                    V = np.delete(V, i, axis=0)
                    N = np.delete(N, i, axis=0)
                    Y.pop(i)
                else:
                    i+=1
            
            # Normalise the data
            D = normalizer(D)
            realP = normalizer(realP)
            V = normalizer(V)
            N = normalizer(N)
            n+=1
            # write data.csv using pandas
            data = {'Dynamic': D, 'rPitch': realP, 'Velocity': V, 'Pitch': N}
            # add lyrics to the data
            for i in range(len(ydict)):
                data[ydict[i]] = [y[i] for y in Y]
            df = pd.DataFrame(data)
            if  num+1-j < 3:
                df.to_csv('testfiles/data.csv', index=False)
            df.to_csv('trainfiles/data'+str(n)+'.csv', index=False)

    except:
        print('Error in '+str(j))
        continue