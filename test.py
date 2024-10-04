import logging
from logging import StreamHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
import os.path as osp
import click
import yaml
import shutil
from torch.utils.tensorboard import SummaryWriter
from munch import Munch
from utils import get_data_path_list
from meldataset import build_dataloader
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert
from models import *
from utils import *
import random
# %%
config = yaml.safe_load(open("Configs/config_ft.yml"))

ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load PL-BERT model
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

# build model
model_params = recursive_munch(config['model_params'])
multispeaker = model_params.multispeaker
device = 'cuda'
args = model_params
from Modules.istftnet import Decoder
# Thiết lập seed để tái tạo
random.seed(42)
torch.manual_seed(42)

# Kích thước batch và sequence
batch_size = 1
seq_len = 100

# Các tham số đầu vào
dim_in = 512
F0_channel = 512
style_dim = 64

# Tạo các tensor đầu vào
asr = torch.randn(batch_size, dim_in, seq_len)
F0_curve = torch.randn(batch_size, seq_len)
N = torch.randn(batch_size, seq_len)
s = torch.randn(batch_size, style_dim)

# Khởi tạo mô hình
decoder = Decoder(
    dim_in=dim_in,
    F0_channel=F0_channel,
    style_dim=style_dim,
    dim_out=80,
    resblock_kernel_sizes=[3, 7, 11],
    upsample_rates=[10, 6],
    upsample_initial_channel=512,
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
    upsample_kernel_sizes=[20, 12],
    gen_istft_n_fft=20,
    gen_istft_hop_size=5
)
# Chuyển sang chế độ eval
decoder.eval()

# Chuyển mô hình sang CPU để đơn giản hóa quá trình xuất ONNX
decoder = decoder.cpu()
asr = asr.cpu()
F0_curve = F0_curve.cpu()
N = N.cpu()
s = s.cpu()

# Hàm wrapper để đảm bảo đầu ra là tuple
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, asr, F0_curve, N, s):
        output = self.decoder(asr, F0_curve, N, s)
        return (output,)

# Wrap mô hình
wrapped_decoder = DecoderWrapper(decoder)

# Xuất sang ONNX
try:
    torch.onnx.export(
        wrapped_decoder,
        (asr, F0_curve, N, s),  # Đầu vào là một tuple chứa các tham số
        "hifigan.onnx",
        input_names=["asr", "F0_curve", "N", "s"],  # Tên cho các input
        output_names=["output"],
        dynamic_axes={'asr': {2: 'seq_len'},
                      'F0_curve': {1: 'seq_len'},
                      'N': {1: 'seq_len'},
                      'output': {2: 'seq_len'}},
        do_constant_folding=True,
        opset_version=11,
        verbose=True
    )
    print("Xuất ONNX thành công!")
except Exception as e:
    print(f"Lỗi khi xuất ONNX: {str(e)}")

# Kiểm tra mô hình ONNX
# import onnx
# onnx_model = onnx.load("hifigan.onnx")
# onnx.checker.check_model(onnx_model)
# print("Kiểm tra mô hình ONNX thành công!")
#
# print(f"\nKích thước đầu vào:")
# print(f"asr: {asr.shape}")
# print(f"F0_curve: {F0_curve.shape}")
# print(f"N: {N.shape}")
# print(f"s: {s.shape}")