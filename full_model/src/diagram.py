import torch
from torchviz import make_dot
from NILMModel import NILMModel

model = NILMModel()
model.load_state_dict(torch.load('/home/hernies/Documents/tfg/full_model/first_test/nilm_model.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
diagram=make_dot(model(dummy_input), params=dict(model.named_parameters()))
diagram.format = 'png'
diagram.directory = '/home/hernies/Documents/tfg/full_model/first_test'
diagram.view()
