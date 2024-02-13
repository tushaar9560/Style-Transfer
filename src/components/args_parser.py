import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="parser for style-transfer")
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="Number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="Batch size for training , default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                  "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str , default="artifacts/styles/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type = str, required=True,
                                  help="path to folder where trained model will be saved")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained model will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="Size of training images, default is 256x256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="Size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help='set it to 1 for running on GPU, 0 for CPU')
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed on training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="Weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type = float, default=1e10,
                                  help="Weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="Learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help = "Number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = parser.parse_args()

    return args