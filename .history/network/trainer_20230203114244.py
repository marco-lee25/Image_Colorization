from .utils import create_loss_meters, update_losses, log_results, visualize
from tqdm import tqdm
from torch import optim, nn
from GAN import build_backbone_unet, MainModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class model_trainer():
    def __init__(self, train_loader, ic, oc, config, display=100):
        self.train_loader = train_loader
        self.epochs = config.epochs
        self.display = display
        self.ic = ic
        self.oc = oc

        self.generator = build_backbone_unet(input_channels=self.ic, output_channels=oc, size=config.image_size_1, layers_to_cut=config.layers_to_cut)
        self.opt = optim.Adam(self.generator.parameters(), lr=config.pretrain_lr)
        self.loss = nn.L1Loss()        
        self.pretrain_generator(self.generator, train_loader, self.opt, self.loss, config.epochs)

        torch.save(self.generator.state_dict(), "res18-unet.pt")
        self.generator.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        self.model = MainModel(config, generator=self.generator)

    def get_model(self):
        return self.model

    def train_model(self):
        data = next(iter(self.train_loader)) 
        for epoch in range(self.epochs):
            loss_meter_dict = create_loss_meters() 
            i=0
            for data in tqdm(self.train_loader):
                self.model.prepare_input(data) 
                self.model.optimize()
                update_losses(self.model, loss_meter_dict, count=data['L'].size(0)) 
                i+=1
                if i % self.display == 0:
                    print(f"\nEpoch {epoch+1}/{self.epochs}")
                    print(f"Iteration {i}/{len(self.train_loader)}")
                    log_results(loss_meter_dict) 
                    visualize(self.model, data, save=False) 

    def save_model(self):
        torch.save(self.model.state_dict(), "main-model.pt")