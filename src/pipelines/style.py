import torch
from torchvision import transforms
from utils import save_image, check_model_exists
import os
import sys
import onnx
import onnx_caffe2.backend
import re
from logger import logging
from exception import CustomException
from components.transformer_net import TransformerNet

class StyleTransfer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda  else "cpu")
        self.style_model = self.load_model(args.model)
        
    def load_model(self, model_path:str):
        try:
            # Load Pytorch model if .pth file
            if model_path.endswith(".pth"):
                logging.info("Loadig pytorch model from %s", model_path)
                style_model = TransformerNet()
                state_dict = torch.load(model_path)
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
                style_model.load_state_dict(state_dict)
                style_model.to(self.device)
                style_model.eval()
                return style_model
            
            # Load ONNX model if .onnx file
            elif model_path.endswith(".onnx"):
                logging.info("Loading ONNX model from %s", model_path)
                assert not self.args.export_onnx
                model = onnx.load(self.args.model)
                prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if self.args.cuda else 'CPU')
                return prepared_backend
            else:
                raise CustomException("Invalid model file format. Supported formats: .pth(PyTorch) or .onnx(Caffe2)", None)
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def stylize(self, content_image_path:str, output_image_path:str):
        
