from utils import create_loss_meters, update_losses, log_results, visualize
from tqdm import nn, tqdm
from torch import optim
from GAN import build_backbone_unet

class model_trainer():
    def __init__(self, model, train_loader, ic, oc, config, display=100):
        self.model = model
        self.train_loader = train_loader
        self.epochs = config.epochs
        self.display = display
        self.ic = ic
        self.oc = oc

        self.generator = build_backbone_unet(input_channels=self.ic, output_channels=oc, size=config.image_size_1)
        self.opt = optim.Adam(self.generator.parameters(), lr=config.pretrain_lr)
        self.loss = nn.L1Loss()        

def train_model(model, train_loader, epochs, display=100):
    data = next(iter(train_loader)) 
    for epoch in range(epochs):
        loss_meter_dict = create_loss_meters() 
        i=0
        for data in tqdm(train_loader):
            model.prepare_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) 
            i+=1
            if i % display == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Iteration {i}/{len(train_loader)}")
                log_results(loss_meter_dict) 
                visualize(model, data, save=False) 