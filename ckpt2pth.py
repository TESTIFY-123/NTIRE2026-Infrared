import torch
import os

folder = "/data2/pl/zh/NTIRE2026/NTIRE2026_infraredSR/model_zoo"

for file in os.listdir(folder):
    if file.endswith(".ckpt"):
        ckpt_path = os.path.join(folder, file)
        pth_path = os.path.splitext(ckpt_path)[0] + ".pth"

        ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")
            new_state_dict[new_key] = v

        torch.save(new_state_dict, pth_path)


        #
        # torch.save(state_dict, pth_path)

        print(f"Converted: {file}")