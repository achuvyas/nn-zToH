import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()


chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}



xs=[]
ys=[]

for w in words[:1]:
    chs=["."] + list(w) + ["."]
    for ch1,ch2, ch3 in zip(chs[1:], chs[2:], chs[3:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        xs.append([ix1,ix2])
        ys.append([ix3])



xs=torch.tensor(xs)
ys=torch.tensor(ys)


print(xs)


xenc = F.one_hot(xs, 27).float()

print(xenc.shape)


W = torch.randn((27, 1))
mutl_val = xenc @ W


print(mutl_val)



   



