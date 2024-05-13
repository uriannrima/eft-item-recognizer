from time import time
from custom_config import Config 
import torch

def train(train_dataloader, optimizer, net, criterion):
    counter = []
    loss_history = [] 
    iteration_number= 0

    prevNum = -1
    
    for epoch in range(0,Config.train_number_epochs):
        begin = time()

        for i, data in enumerate(train_dataloader,0):
            print(i)
            img_anc, img_pos, img_neg,_ = data
            img_anc, img_pos, img_neg = img_anc.cuda(), img_pos.cuda(), img_neg.cuda()
            optimizer.zero_grad()
            output1,output2,output3 = net(img_anc, img_pos , img_neg)
            loss_contrastive = criterion(output1,output2,output3)
            loss_contrastive.backward()
            optimizer.step()
            optimizer.zero_grad()

            # To prevent repetation of epoch
            if i %10 == 0 and prevNum != epoch:
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
                prevNum = epoch

        savePath = './res.pth'
        torch.save(net.state_dict(), savePath)
        print(time()-begin, 's has passed')