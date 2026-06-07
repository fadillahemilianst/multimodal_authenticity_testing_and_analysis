import torch
ck = torch.load("model/best_rawnet_audio_binary.pth", map_location="cpu")
print(type(ck))
print(ck.keys() if isinstance(ck, dict) else "bukan dict - full model")