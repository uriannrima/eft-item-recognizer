import torchvision.models as models
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()


        #self.resnet = models.resnet152(pretrained=True)
        #self.resnet = models.resnet101(pretrained=True)
        self.resnet = models.resnet50(pretrained=True)

        #self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def forward_once(self, x):
        '''
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        #print(output.shape)
        #print(output)
        '''
        #begin = time()
        output = self.resnet(x)
        #print('Time for forward prop: ', time()-begin)

        return output
        

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3