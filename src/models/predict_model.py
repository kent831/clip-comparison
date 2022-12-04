import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu" 
model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
#checkpoint = torch.load("model_cifar10_photo.pt")
#model.load_state_dict(checkpoint['model_state_dict'])

train_data = CIFAR10(download=False,root="./data",transform=preprocess)
test_data = CIFAR10(root="./data",train=False,transform=preprocess)

train_dl = DataLoader(train_data,16,num_workers=2,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,16,num_workers=2,pin_memory=True)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_data.classes]).to(device)


with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    zeroshot_weights = []
    for batch in tqdm(test_dl):        
        images,label = batch       
   
        images= images.to(device)
        texts = text_inputs
        #torch.cat([clip.tokenize(f"a photo of a {test_data.classes[l]}") for l in label]).to(device)
        
        target = label.to(device)
        
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        class_embedding = text_features.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        
        similarity = 100.0 * image_features @ zeroshot_weights
        #values, indices = similarity[0].topk(5)
        acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
 
top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")