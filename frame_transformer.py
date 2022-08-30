import math
import torch
from torch import nn
import torch.nn.functional as F
from frame_quantizer import FrameQuantizer
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False, expansion=expansion)
        self.enc1_transformer = FrameTransformerEncoder(channels, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin, expansion=expansion)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, expansion=expansion)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, expansion=expansion)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, expansion=expansion)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16, expansion=expansion)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, self.max_bin // 32, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec5 = FrameDecoder(channels * 10, channels * 8, self.max_bin // 16, expansion=expansion)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8, expansion=expansion)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4, expansion=expansion)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2, expansion=expansion)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin, expansion=expansion)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.out = nn.Parameter(torch.empty(in_channels, channels))
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        e6 = self.enc6_transformer(self.enc6(e5))
        d5 = self.dec5_transformer(self.dec5(e6, e5), e5)
        d4 = self.dec4_transformer(self.dec4(d5, e4), e4)
        d3 = self.dec3_transformer(self.dec3(d4, e3), e3)
        d2 = self.dec2_transformer(self.dec2(d3, e2), e2)
        d1 = self.dec1_transformer(self.dec1(d2, e1), e1)

        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)    

        return out

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=True):
        super(MultichannelLinear, self).__init__()

        self.weight_pw = None
        if in_features != out_features or positionwise:
            self.weight_pw = nn.Parameter(torch.empty(in_channels, out_features, in_features))
            nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

        self.weight_dw = None
        if in_channels != out_channels or depthwise:
            self.weight_dw = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_dw, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3) 
        
        return x

class FrameNorm(nn.Module):
    def __init__(self, features):
        super(FrameNorm, self).__init__()

        self.norm = nn.LayerNorm(features)

    def __call__(self, x):
        return self.norm(x.transpose(2,3)).transpose(2,3)

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, expansion=2, dropout=0.1):
        super(FrameEncoder, self).__init__()

        self.relu = nn.ReLU()
        self.norm1 = FrameNorm(features)
        self.linear1 = MultichannelLinear(in_channels, out_channels, features, features * 2)
        self.linear2 = MultichannelLinear(out_channels, out_channels, features * 2, features // 2 if downsample else features)
        self.identity = MultichannelLinear(in_channels, out_channels, features, features // 2 if downsample else features, positionwise=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x):
        h = self.norm1(x)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = self.identity(x) + self.dropout(h)
        
        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, upsample=True, expansion=2, dropout=0.1, has_skip=True):
        super(FrameDecoder, self).__init__()
        
        self.relu = nn.ReLU()
        self.norm = FrameNorm(features // 2)
        self.linear1 = MultichannelLinear(in_channels, out_channels, features // 2, features * 2)
        self.linear2 = MultichannelLinear(out_channels, out_channels, features * 2, features)
        self.identity = MultichannelLinear(in_channels, out_channels, features // 2, features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if has_skip:
            self.norm2 = FrameNorm(features)
            self.linear3 = MultichannelLinear(out_channels * 2, out_channels, features, features * 2)
            self.linear4 = MultichannelLinear(out_channels, out_channels, features * 2, features, depthwise=False)
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip=None):
        h = self.norm(x)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = self.identity(x) + self.dropout(h)

        if skip is not None:
            h = self.norm2(torch.cat((x, skip), dim=1))
            h = self.linear4(torch.square(self.relu(self.linear3(h))))
            x = x + h

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, mixed_precision=False):
        super().__init__()

        self.mixed_precision = mixed_precision
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads // 2)

        self.q_norm = FrameNorm(features // num_heads)
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,9), padding=(0,4), bias=False))

        self.k_norm = FrameNorm(features // num_heads)
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,9), padding=(0,4), bias=False))

        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,9), padding=(0,4), bias=False))

        self.out_proj = MultichannelLinear(channels, channels, features, features)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape

        q = self.rotary_embedding.rotate_queries_or_keys(self.q_norm(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_norm(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)        
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)

        return x

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, num_heads=4, dropout=0.1, expansion=4):
        super(FrameTransformerEncoder, self).__init__()

        self.relu = nn.ReLU()

        self.norm1 = FrameNorm(features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion, depthwise=False)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features, depthwise=False)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        z = self.dropout1(z)
        x = x + z

        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        z = self.dropout2(z)
        x = x + z

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, features, num_heads=4, dropout=0.1, expansion=4):
        super(FrameTransformerDecoder, self).__init__()
        
        self.relu = nn.ReLU()

        self.norm1 = FrameNorm(features)
        self.attn1 = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(features)
        self.attn2 = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = FrameNorm(features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion, depthwise=False)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features, depthwise=False)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip):
        z = self.norm1(x)
        z = self.attn1(z)
        z = self.dropout1(z)
        x = x + z

        z = self.norm2(x)
        z = self.attn2(z, mem=skip)
        z = self.dropout2(z)
        x = x + z

        z = self.norm3(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        z = self.dropout3(z)
        x = x + z

        return x

class FrameTransformerDiscriminator(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2):
        super(FrameTransformerDiscriminator, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False, expansion=expansion)
        self.enc1_transformer = FrameTransformerEncoder(channels, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin, expansion=expansion)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, expansion=expansion)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, expansion=expansion)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, expansion=expansion)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16, expansion=expansion)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, self.max_bin // 32, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc7 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32, expansion=expansion)
        self.enc7_transformer = FrameTransformerEncoder(channels * 12, self.max_bin // 64, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc8 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32, expansion=expansion)
        self.enc8_transformer = FrameTransformerEncoder(channels * 12, self.max_bin // 64, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc9 = FrameEncoder(channels * 12, channels * 14, self.max_bin // 64, expansion=expansion)
        self.enc9_transformer = FrameTransformerEncoder(channels * 14, self.max_bin // 128, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc10 = FrameEncoder(channels * 14, channels * 16, self.max_bin // 128, expansion=expansion)
        self.enc10_transformer = FrameTransformerEncoder(channels * 16, self.max_bin // 256, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc11 = FrameEncoder(channels * 16, channels * 18, self.max_bin // 256, expansion=expansion)
        self.enc11_transformer = FrameTransformerEncoder(channels * 18, self.max_bin // 512, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc12 = FrameEncoder(channels * 18, 1, self.max_bin // 512, expansion=expansion)

    def __call__(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        e6 = self.enc6_transformer(self.enc6(e5))
        e7 = self.enc7_transformer(self.enc7(e6))
        e8 = self.enc8_transformer(self.enc8(e7))
        e9 = self.enc9_transformer(self.enc9(e8))
        e10 = self.enc10_transformer(self.enc10(e9))
        e11 = self.enc11_transformer(self.enc11(e10))
        e12 = self.enc12(e11)

        return e12

class VQFrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2, num_embeddings=2048):
        super(VQFrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False, expansion=expansion)
        self.enc1_transformer = FrameTransformerEncoder(channels, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin, expansion=expansion)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, expansion=expansion)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, expansion=expansion)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, expansion=expansion)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.fq = FrameQuantizer(channels * 8, self.max_bin // 16, num_embeddings)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8, expansion=expansion, has_skip=False)
        self.dec4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4, expansion=expansion, has_skip=False)
        self.dec3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2, expansion=expansion, has_skip=False)
        self.dec2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin, expansion=expansion, has_skip=False)
        self.dec1_transformer = FrameTransformerEncoder(channels * 1, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.out = nn.Parameter(torch.empty(in_channels, channels)) if in_channels != channels else None
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        q, ql, indices = self.fq(e5)
        d4 = self.dec4_transformer(self.dec4(q))
        d3 = self.dec3_transformer(self.dec3(d4))
        d2 = self.dec2_transformer(self.dec2(d3))
        d1 = self.dec1_transformer(self.dec1(d2))
 
        if self.out is not None:
            out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return torch.sigmoid(out), ql