from utils import create_loss_meters, update_losses, log_results, visualize
from tqdm import tqdm
from torch import optim, nn
from GAN import build_backbone_unet, MainModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.pretrain_generator(self.generator, train_loader, self.opt, self.loss, config.epochs)

        torch.save(self.generator.state_dict(), "res18-unet.pt")
        self.generator.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        self.model = MainModel(generator=self.generator)
        train_model(model, train_loader, Config.epochs)

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