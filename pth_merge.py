import torch
from models.team03_HAT_PFT import MultiModelAverage  # 这里改成你的实际文件路径

def save_ensemble_pth(
    hat1_path,
    hat2_path,
    pft1_path,
    pft2_path,
    save_path,
    hat_weight=0.25,
    hat_second_weight=0.25,
    pft_weight=0.25,
    pft_second_weight=0.25
):
    # 先创建集成模型
    model = MultiModelAverage(
        hat_weight=hat_weight,
        hat_second_weight=hat_second_weight,
        pft_weight=pft_weight,
        pft_second_weight = pft_second_weight
    )

    # 分别读取三个模型权重
    hat1_state = torch.load(hat1_path, map_location="cpu")
    hat2_state = torch.load(hat2_path, map_location="cpu")
    pft1_state = torch.load(pft1_path, map_location="cpu")
    pft2_state = torch.load(pft2_path, map_location="cpu")

    # 加载到对应子模块
    model.hat_1.load_state_dict(hat1_state, strict=True)
    model.hat_2.load_state_dict(hat2_state, strict=True)
    model.pft1.load_state_dict(pft1_state, strict=True)
    model.pft2.load_state_dict(pft2_state, strict=True)


    # 保存整个 ensemble 的 state_dict
    torch.save(model.state_dict(), save_path)
    print(f"Ensemble model saved to: {save_path}")


if __name__ == "__main__":
    save_ensemble_pth(
        hat1_path="model_zoo/team03_HAT_PFT/hat_ssim.pth",
        hat2_path="model_zoo/team03_HAT_PFT/sr_hat.pth",
        pft1_path="model_zoo/team03_HAT_PFT/pft.pth",
        pft2_path="model_zoo/team03_HAT_PFT/pft_l1.pth",
        save_path="ensemble_model.pth",
        hat_weight=0.25,
        hat_second_weight=0.25,
        pft_weight=0.25,
        pft_second_weight=0.25
    )