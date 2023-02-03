from utils import create_loss_meters, update_losses, log_results, visualize
from tqdm import tqdm

class model_trainer():
    def __init__(self, model, train_loader, epochs, ic, oc, display=100):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.display = display
        self.ic = ic
        self.oc = oc



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