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


train_data = CIFAR10(download=False,root="./data",transform=preprocess)
test_data = CIFAR10(root="./data",train=False,transform=preprocess)

train_dl = DataLoader(train_data,16,num_workers=2,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,16,num_workers=2,pin_memory=True)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()
else :
    clip.model.convert_weights(model) 


loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

#train
EPOCH = 20
history = []
for epoch in tqdm(range(EPOCH)):
   
    epoch_loss = []
    for batch in tqdm(train_dl):
        optimizer.zero_grad()

        images,label = batch 
        texts = torch.cat([clip.tokenize(f"a photo of a {test_data.classes[l]}") for l in label])
        
        images= images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            
        epoch_loss.append(total_loss.item())
        total_loss.backward()
        
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
    loss = np.mean(epoch_loss)
    print("epoch: {}      loss: {}".format(epoch, loss))
    if len(history) >= 1:
        if loss > history[-1]:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"model_cifar10_photo.pt")
            break
    history.append(loss)

