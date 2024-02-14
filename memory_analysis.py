import torch
from Model import CLMP
import numpy as np

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)


model = CLMP()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)
input_data = torch.randn(64, 180,512).cuda()
media_mask = np.where(np.all(input_data.cpu().numpy() == 0, axis=2), True, False)
media_mask = torch.from_numpy(media_mask).cuda()
model = model.cuda()

with torch.cuda.device(0):
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    print(input_data.device, media_mask.device)
    output = model(input_data, input_data, media_mask)

    memory_stats = torch.cuda.memory_stats(torch.cuda.current_device())
    max_memory_allocated = memory_stats['allocated_bytes.all.peak']
    max_memory_cached = memory_stats['active_bytes.all.peak']

print(f"Max Memory Allocated: {max_memory_allocated / 1024**2:.2f} MB")
print(f"Max Memory Cached: {max_memory_cached / 1024**2:.2f} MB")