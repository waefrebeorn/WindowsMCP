import math
import os # Added missing import
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

from .utils import DEFAULT_DEVICE # Corrected relative import


class logFbankCal(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ):
        super().__init__()
        self.fbankCal = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            n_mels=n_mels,
        )

    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2) # PyTorch 2.x: mean(dim=2)
        return out


class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        # self.out_dim = in_planes * 8 * outmap_size * 2 # This variable is defined but not used in __init__

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        # Storing out_dim for potential external use, if needed by other parts of a model.
        self.out_dim_calculated = in_planes * 8 * outmap_size * 2


    def forward(self, x):
        # x shape: [batch, channels, freq_bins, time_frames] e.g. [B, 512, 10, T/8] if in_planes=64, expansion=8
        # Reshape for Conv1d: [batch, features, time_frames]
        # features = channels * freq_bins
        x = x.reshape(x.size()[0], -1, x.size()[-1]) # x: [B, C*F_bins, T_frames]
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1) # Flatten
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1): # block_id not used
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SimAM applied to the output of conv2 before adding shortcut
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        # X shape: [B, C, H, W] for 2D conv, or [B, C, L] for 1D conv
        # n is number of spatial/temporal elements per channel
        if X.dim() == 4: # 2D features
            n = X.shape[2] * X.shape[3] - 1
            mean_dims = [2, 3]
        elif X.dim() == 3: # 1D features
            n = X.shape[2] - 1
            mean_dims = [2]
        else:
            raise ValueError(f"Unsupported input dimension for SimAM: {X.dim()}")

        if n < 0: n = 0 # Avoid negative n if spatial/temporal dim is 1

        # d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2) # Original for 2D
        d = (X - X.mean(dim=mean_dims, keepdim=True)).pow(2)

        # v = d.sum(dim=[2, 3], keepdim=True) / n # Original for 2D
        # Handle n=0 case for division
        v = d.sum(dim=mean_dims, keepdim=True) / (n if n > 0 else 1)

        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module): # Standard ResNet BasicBlock
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1): # block_id not used
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module): # Standard ResNet Bottleneck
    expansion = 4

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1): # block_id not used
        super(Bottleneck, self).__init__()
        # ConvLayer and NormLayer are passed but nn.Conv2d and nn.BatchNorm2d are hardcoded
        # This should be made consistent if this block is intended for flexible Conv/Norm types.
        # Assuming 2D for now based on hardcoding.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True) # Added ReLU

        self.shortcut = nn.Sequential() # Renamed from downsample for consistency with ResNet terminology
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        # Original used F.relu, but self.relu is defined. Using self.relu.
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes_init, block, num_blocks, in_ch=1, feat_dim="2d", **kwargs): # Renamed in_planes to in_planes_init
        super(ResNet, self).__init__()
        if feat_dim == "1d":
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim == "2d":
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim == "3d":
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            # print("error") # Should raise error
            raise ValueError(f"Unsupported feature dimension for ResNet: {feat_dim}")

        self.in_planes = in_planes_init # This is current in_planes for layer construction

        self.conv1 = self.ConvLayer(in_ch, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1) # block_id removed
        self.layer2 = self._make_layer(block, self.in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_planes * 8, num_blocks[3], stride=2)
        # Note: self.in_planes is modified by _make_layer. If using fixed planes for each layer,
        # it should be passed directly, e.g., self.in_planes for layer1, planes_init*2 for layer2, etc.
        # The current way means _make_layer updates self.in_planes based on block.expansion.
        # For BasicBlock (expansion=1), self.in_planes effectively becomes `planes` argument.
        # For SimAMBasicBlock (expansion=1), this also holds.
        # If planes for layer2 should be in_planes_init*2, then pass that.
        # The original code for _make_layer:
        # self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        # self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        # This implies `planes` argument to _make_layer is the target output planes for that stage.
        # And self.in_planes is the input to the *first block of that stage*.
        # Let's adjust _make_layer call to reflect this more clearly.
        # No, the original ResNet293 calls ResNet(in_planes, SimAMBasicBlock, [10,20,64,3]).
        # Here in_planes is the initial number of planes after conv1.
        # So, layer1 gets `planes=in_planes_init`.
        # layer2 gets `planes=in_planes_init*2`. This seems correct.
        # The `self.in_planes` tracks the output channels of the previous layer/block.


    def _make_layer(self, block, planes, num_blocks, stride): # block_id removed
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, current_stride in enumerate(strides): # Renamed stride to current_stride
            # For the first block of a stage, self.in_planes is from previous stage.
            # For subsequent blocks, self.in_planes is planes * block.expansion.
            # block_id was not used by SimAMBasicBlock or BasicBlock.
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion # Update in_planes for the next block/layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNet293(in_planes: int, **kwargs): # Typically in_planes for ResNet is after first conv, e.g. 64
    # The call ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs) means:
    # in_planes for ResNet's conv1.
    # Layer 1: 10 blocks, output planes = in_planes
    # Layer 2: 20 blocks, output planes = in_planes * 2
    # Layer 3: 64 blocks, output planes = in_planes * 4
    # Layer 4: 3 blocks,  output planes = in_planes * 8
    # This seems like very deep network, especially layer3.
    # Standard ResNet (e.g. ResNet34) uses [3,4,6,3] blocks.
    # Assuming the numbers [10,20,64,3] are correct as per model design.
    # kwargs typically include feat_dim="2d", in_ch=1 for spectrograms
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
    def __init__(
        self,
        in_planes_resnet: int = 64, # Planes for the ResNet backbone's first conv output
        embd_dim: int = 256,    # Output embedding dimension
        acoustic_dim: int = 80, # Number of Mel bins in input spectrogram
        featCal=None,           # Feature calculator (e.g. logFbankCal)
        dropout: float = 0,
        **kwargs, # Passed to ResNet293 (e.g. in_ch, feat_dim)
    ):
        super(ResNet293_based, self).__init__()
        self.featCal = featCal if featCal is not None else logFbankCal(n_mels=acoustic_dim) # Default if None

        # ResNet293 will have feat_dim="2d", in_ch=1 (for single channel spectrogram) by default
        # kwargs for ResNet293 might include these if different.
        # The `in_planes_resnet` is the number of channels after ResNet's conv1.
        self.front = ResNet293(in_planes=in_planes_resnet, feat_dim=kwargs.get("feat_dim", "2d"), in_ch=kwargs.get("in_ch",1))

        block_expansion = SimAMBasicBlock.expansion # This is 1

        # ASP input planes: output planes of ResNet's layer4.
        # ResNet's layer4 has `planes = in_planes_resnet * 8`.
        # So ASP input planes should be `in_planes_resnet * 8 * block_expansion`.
        asp_in_planes = in_planes_resnet * 8 * block_expansion

        # ASP's `acoustic_dim` argument is for its internal calculation of `outmap_size`.
        # It's related to the size of the frequency dimension of the feature map *entering* ASP.
        # ResNet293 output shape: [B, C_out, F_out, T_out]
        # C_out = asp_in_planes
        # F_out = acoustic_dim / 8 (due to striding in ResNet, if acoustic_dim was input H to ResNet)
        # T_out = Time / 8
        # The ASP module in the original code expects `acoustic_dim` to be the H of the input to ResNet.
        # Let's assume `acoustic_dim` passed to ResNet293_based is this input height.
        # ResNet downsamples H and W by 2 three times (layer2, layer3, layer4 strides). So H_out = H_in / 8.
        # This `acoustic_dim` for ASP should be H_in (original Mel bins).
        self.pooling = ASP(asp_in_planes, acoustic_dim=acoustic_dim) # Pass original acoustic_dim (mel_bins)

        # Bottleneck layer input dim comes from ASP's output
        # ASP's self.out_dim_calculated = asp_in_planes * (acoustic_dim / 8) * 2
        # This is: (ResNet_L4_channels) * (FreqBins_at_ASP_input / 8) * 2
        # No, ASP's self.out_dim_calculated is `in_planes * 8 * outmap_size * 2` where `in_planes` is its own input planes.
        # So, `asp_in_planes * (acoustic_dim / 8) * 2`.
        # This is `(in_planes_resnet * 8 * 1) * (acoustic_dim / 8) * 2`
        # ` = in_planes_resnet * acoustic_dim * 2`
        # This seems to be the intended input dimension for the final linear layer.
        self.bottleneck = nn.Linear(self.pooling.out_dim_calculated, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None # Check dropout > 0

    def forward(self, x): # x is raw waveform [B, L_wav] or [L_wav]
        x = self.featCal(x) # x becomes [B, Mel_bins, T_frames]
        # ResNet293 (2D conv) expects [B, C_in, H, W]. Here C_in=1.
        x = self.front(x.unsqueeze(dim=1)) # x becomes [B, 1, Mel_bins, T_frames] -> ResNet output [B, C_out, Mel_bins/8, T_frames/8]
        x = self.pooling(x) # ASP output [B, pooling.out_dim_calculated]
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x) # Final embedding [B, embd_dim]
        return x


# ECAPA-TDNN related classes (SEModule, Bottle2neck, ECAPA_TDNN) seem unrelated to ResNet293 speaker model.
# They are present in the original file but might be for a different speaker encoder.
# For now, I will include them as they were in the original zonos/speaker_cloning.py.

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # Removed in original
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation # kernel_size and dilation must be provided
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i] # Summing feature maps? Original ECAPA splits and concats. This is different.
                                 # Let's assume this is intended as per the provided code.
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1) # Concatenating the last split part

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C, featCal=None, acoustic_dim=80): # Added acoustic_dim for featCal default
        super(ECAPA_TDNN, self).__init__()
        self.featCal = featCal if featCal is not None else logFbankCal(n_mels=acoustic_dim)
        self.conv1 = nn.Conv1d(acoustic_dim, C, kernel_size=5, stride=1, padding=2) # Input channels = acoustic_dim (e.g. 80 mels)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        # Bottle2neck requires kernel_size and dilation. Using typical ECAPA values.
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)

        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1) # Concatenation of 3 layers of C channels

        # Attentive Statistics Pooling for ECAPA
        # Input to attention is concatenation of x, mean(x), std(x)
        # x shape from layer4: [B, 1536, T]
        # So, global_x input channels = 1536 * 3 = 4608
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1), # Original had 4608 (1536*3)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1), # Output channels match x (1536) for weighted sum
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(1536 * 2) # mu and sg concatenation, so 1536*2
        self.fc6 = nn.Linear(1536 * 2, 192) # Output embedding dim 192
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x): # x is raw waveform
        x = self.featCal(x) # x: [B, Mel_bins, T_frames]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1) # Residual connection from x to x2 input
        x3 = self.layer3(x + x1 + x2) # Residual connection from x+x1 to x3 input

        x_concat = torch.cat((x1, x2, x3), dim=1) # [B, 3*C, T]
        x = self.layer4(x_concat) # [B, 1536, T]
        x = self.relu(x)

        t = x.size()[-1] # Number of time frames

        # Attentive Statistics Pooling part
        mean_x = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t)
        std_x = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        global_x = torch.cat((x, mean_x, std_x), dim=1) # [B, 1536*3, T]

        w = self.attention(global_x) # w: [B, 1536, T] (attention weights)

        mu = torch.sum(x * w, dim=2) # [B, 1536]
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4)) # [B, 1536]

        x = torch.cat((mu, sg), 1) # [B, 1536*2]
        x = self.bn5(x)
        x = self.fc6(x) # [B, 192]
        x = self.bn6(x)

        return x


class SpeakerEmbedding(nn.Module): # Wrapper for ResNet293_based model
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: str = DEFAULT_DEVICE, acoustic_dim=80):
        super().__init__()
        self._device_str = device # Store original device string
        self.target_device = torch.device(device) # Actual torch device

        # Model parameters will be on self.target_device after .to(self.target_device)
        self.model = ResNet293_based(acoustic_dim=acoustic_dim) # Pass acoustic_dim

        if os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True) # Load to CPU first
                self.model.load_state_dict(state_dict)
                print(f"INFO: SpeakerEmbedding model loaded from {ckpt_path}")
            except Exception as e:
                print(f"WARNING: Failed to load SpeakerEmbedding checkpoint from {ckpt_path}: {e}. Model is initialized with random weights.")
        else:
            print(f"WARNING: SpeakerEmbedding checkpoint not found at {ckpt_path}. Model is initialized with random weights.")

        self.model.to(self.target_device) # Move model to target device
        self.model.featCal.to(self.target_device) # Ensure featCal is also on target device if it has params

        self.requires_grad_(False) # Same as self.model.requires_grad_(False)
        self.eval() # Same as self.model.eval()

    @property
    def dtype(self): # Dtype of the model parameters
        return next(self.model.parameters()).dtype

    @property # Make device a property to get the actual model device
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


    @cache # Cache resampler per original sample rate
    def _get_resampler(self, orig_sample_rate: int, target_sample_rate: int = 16_000) -> torchaudio.transforms.Resample:
        # Resampler needs to be on the same device as the tensor it processes.
        # Or, tensor moved to resampler's device. Let's create resampler on target_device.
        return torchaudio.transforms.Resample(orig_sample_rate, target_sample_rate).to(self.target_device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # wav: [B, L] or [L]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0) # Add batch dim: [1, L]
        if wav.shape[0] > 1 and wav.ndim == 2 : # Multi-channel audio in batch, e.g. [B, Channels, L]
            wav = wav.mean(1) # Average channels -> [B, L]
        elif wav.ndim > 2 : # E.g. [B, C, L]
             wav = wav.mean(1) # Average channels

        # Ensure wav is on the correct device before resampling
        wav = wav.to(self.target_device)

        if sample_rate != 16000: # Target sample rate for logFbankCal is 16k
            resampler = self._get_resampler(sample_rate, 16000)
            wav = resampler(wav)
        return wav # wav is now [B, L_resampled] on self.target_device

    def forward(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor: # wav: Host tensor
        # Prepare input handles device placement and resampling
        prepared_wav = self.prepare_input(wav, sample_rate) # Now [B, L_resampled] on self.target_device
        # Model expects [B, L_resampled] and is on self.target_device
        embedding = self.model(prepared_wav) # Output [B, embd_dim]
        return embedding # Returns on self.target_device


class SpeakerEmbeddingLDA(nn.Module): # Wrapper for SpeakerEmbedding + LDA
    def __init__(self, device: str = DEFAULT_DEVICE, acoustic_dim_for_spk_model=80):
        super().__init__()
        self._device_str = device
        self.target_device = torch.device(device)

        # Download paths for models
        # These should be configurable or use constants if they are fixed.
        repo_id_spk_emb = "Zyphra/Zonos-v0.1-speaker-embedding"
        spk_model_filename = "ResNet293_SimAM_ASP_base.pt"
        lda_model_filename = "ResNet293_SimAM_ASP_base_LDA-128.pt"

        try:
            spk_model_path = hf_hub_download(repo_id=repo_id_spk_emb, filename=spk_model_filename)
        except Exception as e:
            print(f"ERROR: Failed to download speaker model '{spk_model_filename}' from '{repo_id_spk_emb}': {e}")
            spk_model_path = None # Will cause SpeakerEmbedding to init randomly

        try:
            lda_spk_model_path = hf_hub_download(repo_id=repo_id_spk_emb, filename=lda_model_filename)
        except Exception as e:
            print(f"ERROR: Failed to download LDA model '{lda_model_filename}' from '{repo_id_spk_emb}': {e}")
            lda_spk_model_path = None

        # Instantiate the base speaker embedding model
        # Pass the target device and acoustic_dim
        self.speaker_model = SpeakerEmbedding(ckpt_path=spk_model_path, device=device, acoustic_dim=acoustic_dim_for_spk_model)

        # LDA layer
        # Determine in_features from speaker_model's output (embd_dim)
        # ResNet293_based default embd_dim is 256. LDA model file implies LDA output is 128.
        # So LDA input must be 256.
        lda_in_features = self.speaker_model.model.bottleneck.out_features # Should be 256
        lda_out_features = 128 # From filename "LDA-128"

        self.lda = nn.Linear(lda_in_features, lda_out_features, bias=True) # Default dtype float32

        if lda_spk_model_path and os.path.exists(lda_spk_model_path):
            try:
                lda_sd = torch.load(lda_spk_model_path, map_location="cpu", weights_only=True) # Load to CPU
                self.lda.load_state_dict(lda_sd)
                print(f"INFO: LDA model loaded from {lda_spk_model_path}")
            except Exception as e:
                print(f"WARNING: Failed to load LDA state_dict from {lda_spk_model_path}: {e}. LDA is random.")
        else:
            print(f"WARNING: LDA model checkpoint not found or not downloaded. LDA is random.")

        self.lda.to(self.target_device) # Move LDA to target device

        self.requires_grad_(False)
        self.eval()

    @property # Make device a property to get the actual model device
    def device(self) -> torch.device:
        # Both speaker_model and lda should be on the same target_device
        return self.target_device


    def forward(self, wav: torch.Tensor, sample_rate: int): # wav: Host tensor
        # speaker_model handles device placement of wav and its own execution
        emb_orig = self.speaker_model(wav, sample_rate) # emb_orig is on self.target_device

        # LDA input needs to be float32 as per original code's lda layer definition
        emb_lda_input = emb_orig.to(torch.float32)
        emb_lda_output = self.lda(emb_lda_input) # emb_lda_output is on self.target_device

        return emb_orig, emb_lda_output # Both tensors are on self.target_device
