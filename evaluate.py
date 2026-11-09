"""Main Trans-TCDM sampling and DWI/FA evaluation."""

import os

import argparse
import json
from importlib import import_module
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from time import time
import h5py

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.segment.mask import median_otsu
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from diffusion import GaussianDiffusion_ddim
from model import UNet


def resolve_subject_ids(test_file: str, provided_ids=None):
    """从h5文件获取subject列表"""
    if provided_ids:
        return provided_ids
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试数据文件不存在: {test_file}")
    with h5py.File(test_file, "r") as f:
        subjects = sorted(list(f.keys()))
    if not subjects:
        raise ValueError(f"测试数据文件 {test_file} 中没有找到任何subject键")
    print(f"从h5文件中找到 {len(subjects)} 个对象: {subjects}")
    return subjects


def cal_psnr(img1, img2):
    """计算PSNR"""
    return psnr(img1, img2, data_range=1)


def cal_ssim(img1, img2):
    """计算SSIM"""
    return ssim(img1, img2, channel_axis=3, data_range=1)


def cal_FA_with_mask(hr_data, sr_data, bvals, bvecs):
    """计算FA图，使用HR的mask来计算SR的FA"""
    hr_maskdata, mask = median_otsu(hr_data, vol_idx=range(1, 31), median_radius=3,
                                     numpass=1, autocrop=False, dilate=2)
    sr_maskdata = sr_data * mask[..., None]
    
    gtab = gradient_table(bvals, bvecs=bvecs)
    tenmodel = dti.TensorModel(gtab)
    
    tenfit_hr = tenmodel.fit(hr_maskdata)
    FA_hr = fractional_anisotropy(tenfit_hr.evals)
    FA_hr[np.isnan(FA_hr)] = 0
    
    tenfit_sr = tenmodel.fit(sr_maskdata)
    FA_sr = fractional_anisotropy(tenfit_sr.evals)
    FA_sr[np.isnan(FA_sr)] = 0
    
    brain_mask = FA_hr > 0.01
    FA_hr_masked = FA_hr * brain_mask
    FA_sr_masked = FA_sr * brain_mask
    
    return FA_hr_masked, FA_sr_masked, FA_hr, FA_sr


def evaluate(modelConfig):
    """主测试函数 - 使用mask提高精度"""
    EvalDataset = import_module(f"datasets.{modelConfig['dataset']}").EvalDataset
    torch.cuda.set_device(modelConfig['output_device'])
    device = torch.device(modelConfig["device"])
    print(f"Using device: {device}")

    # 加载V4模型
    model = UNet(
        T=modelConfig["T"], 
        num_labels=10, 
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], 
        dropout=modelConfig["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    print(f"Model architecture: V4 - Transformer Encoder")
    
    model = nn.DataParallel(model.cuda(), device_ids=modelConfig['gpus'], output_device=modelConfig['gpus'][0])
    
    # 加载best.pth权重
    ckpt_path = modelConfig["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded {ckpt_path}, epoch: {ckpt['epoch']}")
    model.eval()

    # 创建采样器
    sampler = GaussianDiffusion_ddim(
        model, 
        modelConfig["beta_1"], 
        modelConfig["beta_T"], 
        image_size_x=modelConfig['img_size_x'],
        image_size_y=modelConfig['img_size_y'], 
        batch_size=modelConfig['batch_size'],
        channels=modelConfig['channel'],
        timesteps=modelConfig['T'], 
        ddim_timesteps=modelConfig['ddim_timesteps'], 
        device=device,
        U=modelConfig['U']
    ).to(device)

    # 测试数据ID列表
    num_list = resolve_subject_ids(
        modelConfig['test_file_z'],
        modelConfig.get("subjects")
    )
    
    # 存储指标
    processed_ids = []
    psnr_list = []
    ssim_list = []
    psnr_FA_list = []
    ssim_FA_list = []

    result_dir = modelConfig['result_dir']
    os.makedirs(result_dir, exist_ok=True)

    data_dir = modelConfig['data_dir']
    affine_dir = modelConfig['affine_dir']

    start_idx = int(modelConfig.get('start_idx', 0))
    skip_ids = set(modelConfig.get('skip_ids', []))
    force_resample = bool(modelConfig.get('force_resample', False))

    with torch.no_grad():
        for idx, sub_id in enumerate(num_list):
            if idx < start_idx:
                continue
            if sub_id in skip_ids:
                print(f"Skipping subject: {sub_id}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing subject: {sub_id} ({idx+1}/{len(num_list)})")
            print(f"{'='*60}")

            # 加载mask文件 - 关键：使用mask提高精度
            mask_file = os.path.join(data_dir, f'{sub_id}_mask.nii.gz')
            if os.path.exists(mask_file):
                mask, _ = load_nifti(mask_file)
                mask = mask.astype(np.int32)
                print(f"Loaded mask: {mask_file}, shape: {mask.shape}")
            else:
                mask = None
                print(f"Warning: mask file not found: {mask_file}")

            # 加载测试数据
            test_file = modelConfig['test_file_z']
            eval_set = EvalDataset(test_file, sub_id)
            loader = DataLoader(eval_set, batch_size=modelConfig['batch_size'])

            sr_file_out = os.path.join(result_dir, f'{sub_id}_sr_best_v4.nii.gz')
            sr_affine = None
            sr_data = None

            full_img_z = []
            full_hr_z = []
            
            start_time = time()

            if (not force_resample) and os.path.exists(sr_file_out):
                sr_data, sr_affine = load_nifti(sr_file_out)
                print(f"Found existing SR file, skip sampling: {sr_file_out}")

                target_hr_h = modelConfig["img_size_x"]
                target_hr_w = modelConfig["img_size_y"]
                for _, (_, hr, _) in enumerate(loader):
                    hr = hr.float()
                    hr = F.interpolate(hr, size=(target_hr_h, target_hr_w), mode='bilinear', align_corners=False)
                    full_hr_z.append(hr[..., None])
                full_hr_z = torch.cat(full_hr_z, dim=-1)[0]
                full_hr_z = full_hr_z.permute(1, 2, 3, 0)
                hr_data = full_hr_z.cpu().numpy()
            else:
                for i, (lr, hr, fibre_density) in enumerate(loader):
                    hr = hr.to(device)
                    lr = lr.to(device)
                    fibre_density = fibre_density.to(device)
                    if fibre_density.dim() == 3:
                        fibre_density = fibre_density.unsqueeze(1)

                    target_lr_h = modelConfig["img_size_x"] // 2
                    target_lr_w = modelConfig["img_size_y"] // 2
                    target_hr_h = modelConfig["img_size_x"]
                    target_hr_w = modelConfig["img_size_y"]

                    lr = F.interpolate(lr, size=(target_lr_h, target_lr_w), mode='bilinear', align_corners=False)
                    hr = F.interpolate(hr, size=(target_hr_h, target_hr_w), mode='bilinear', align_corners=False)
                    fibre_density = F.interpolate(fibre_density, size=(target_hr_h, target_hr_w), mode='bilinear', align_corners=False)

                    noisyImage = torch.randn(
                        size=[modelConfig["batch_size"], 30, modelConfig["img_size_x"], modelConfig["img_size_y"]],
                        device=device
                    )

                    sampledImgs = []
                    for ii in range(30):
                        print(f'-{sub_id}--z方向---第{i}切片,第{ii}通道--------')
                        noisyImage_single = noisyImage[:, ii:ii+1, :, :]
                        labels_single = lr[:, ii:ii+1, :, :]
                        sampledImgs.append(sampler(noisyImage_single, labels_single, fibre_density))

                    sampledImgs = torch.cat(sampledImgs, dim=1)
                    full_img_z.append(sampledImgs[..., None])
                    full_hr_z.append(hr[..., None])
                    print(f'z方向切片{i}完成，尺寸：{sampledImgs.shape}')

                full_img_z = torch.cat(full_img_z, dim=-1)[0]
                full_hr_z = torch.cat(full_hr_z, dim=-1)[0]

                full_img_z = full_img_z.permute(1, 2, 3, 0)
                full_hr_z = full_hr_z.permute(1, 2, 3, 0)

                end_time = time()
                print(f'Subject {sub_id} 耗时：{end_time - start_time:.2f}s')

                sr_data = full_img_z.cpu().numpy()
                hr_data = full_hr_z.cpu().numpy()

                # 保存原始SR结果
                affine_path = os.path.join(affine_dir, f'{sub_id}_data_affine')
                data_affine = np.loadtxt(affine_path, dtype=np.float32)
                save_nifti(sr_file_out, sr_data, data_affine)
                sr_affine = data_affine
                print(f'{sub_id} - SR已保存: {sr_file_out}')

            # 处理异常值
            sr_data[np.isnan(sr_data)] = 0.0
            sr_data[np.isinf(sr_data)] = 0.0
            sr_data = np.clip(sr_data, 0, 1)

            # 关键：应用mask提高精度
            if mask is not None:
                # 需要调整mask尺寸以匹配sr_data
                src_x, src_y, src_z = mask.shape
                target_slices = sr_data.shape[2]  # z方向切片数
                z_start = src_z // 2 - target_slices // 2
                z_end = z_start + target_slices
                
                # 裁剪mask到中间切片
                mask_cropped = mask[:, :, z_start:z_end]
                
                # 如果mask的x,y尺寸与sr_data不同，需要resize
                if mask_cropped.shape[0] != sr_data.shape[0] or mask_cropped.shape[1] != sr_data.shape[1]:
                    from scipy.ndimage import zoom
                    zoom_factors = (sr_data.shape[0] / mask_cropped.shape[0],
                                   sr_data.shape[1] / mask_cropped.shape[1],
                                   1.0)
                    mask_cropped = zoom(mask_cropped, zoom_factors, order=0)
                
                # 应用mask到sr_data和hr_data的每个通道
                for i in range(sr_data.shape[3]):
                    sr_data[:, :, :, i] = sr_data[:, :, :, i] * mask_cropped
                    hr_data[:, :, :, i] = hr_data[:, :, :, i] * mask_cropped
                print(f'{sub_id} - 已应用mask到SR和HR数据')

            # 计算PSNR和SSIM
            p = cal_psnr(sr_data, hr_data)
            s = cal_ssim(sr_data, hr_data)
            psnr_list.append(p)
            ssim_list.append(s)
            processed_ids.append(sub_id)
            print(f'{sub_id} - PSNR: {p:.4f}, SSIM: {s:.4f}')

            # 计算FA
            b0_file = os.path.join(data_dir, f'{sub_id}_b0.nii.gz')
            bval_file = os.path.join(data_dir, f'{sub_id}_bvals.txt')
            bvec_file = os.path.join(data_dir, f'{sub_id}_bvecs.txt')
            bval_b0_file = os.path.join(data_dir, f'{sub_id}_bvals_b0.txt')
            bvec_b0_file = os.path.join(data_dir, f'{sub_id}_bvecs_b0.txt')

            if not all(os.path.exists(f) for f in [b0_file, bval_file, bvec_file, bval_b0_file, bvec_b0_file]):
                print(f'{sub_id} - 缺少DTI辅助文件，跳过FA计算')
                psnr_FA_list.append(np.nan)
                ssim_FA_list.append(np.nan)
                continue

            b0_data, _ = load_nifti(b0_file)
            if b0_data.ndim == 4 and b0_data.shape[-1] == 1:
                b0_data = b0_data[..., 0]
            bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
            bvals_b0, bvecs_b0 = read_bvals_bvecs(bval_b0_file, bvec_b0_file)
            
            bvals = np.concatenate((bvals_b0[0:1], bvals), axis=0)
            bvecs = np.concatenate((bvecs_b0[0:1], bvecs), axis=0)

            # 取b0的中间切片
            src_z = b0_data.shape[2]
            target_slices = sr_data.shape[2]
            z_start = src_z // 2 - target_slices // 2
            z_end = z_start + target_slices
            b0_middle = b0_data[:, :, z_start:z_end]
            
            if b0_middle.ndim == 3:
                b0_for_concat = b0_middle[..., None]
            else:
                b0_for_concat = b0_middle

            # 调整b0尺寸以匹配sr_data
            if b0_for_concat.shape[0] != sr_data.shape[0] or b0_for_concat.shape[1] != sr_data.shape[1]:
                from scipy.ndimage import zoom
                zoom_factors = (sr_data.shape[0] / b0_for_concat.shape[0],
                               sr_data.shape[1] / b0_for_concat.shape[1],
                               1.0, 1.0)
                b0_for_concat = zoom(b0_for_concat, zoom_factors, order=1)

            hr_with_b0 = np.concatenate((b0_for_concat, hr_data), axis=3)
            sr_with_b0 = np.concatenate((b0_for_concat, sr_data), axis=3)

            print(f'{sub_id} - 计算FA...')
            FA_hr_masked, FA_sr_masked, FA_hr, FA_sr = cal_FA_with_mask(hr_with_b0, sr_with_b0, bvals, bvecs)

            FA_hr_masked = FA_hr_masked[..., None]
            FA_sr_masked = FA_sr_masked[..., None]

            p_FA = cal_psnr(FA_sr_masked, FA_hr_masked)
            s_FA = cal_ssim(FA_sr_masked, FA_hr_masked)
            psnr_FA_list.append(p_FA)
            ssim_FA_list.append(s_FA)
            print(f'{sub_id} - FA PSNR: {p_FA:.4f}, FA SSIM: {s_FA:.4f}')

            # 保存FA结果
            if sr_affine is None:
                affine_path = os.path.join(affine_dir, f'{sub_id}_data_affine')
                sr_affine = np.loadtxt(affine_path, dtype=np.float32)
            save_nifti(os.path.join(result_dir, f'{sub_id}_FA_sr_best_v4.nii.gz'), FA_sr.astype(np.float32), sr_affine)
            print(f'{sub_id} - FA已保存')

    # 计算均值和标准差
    psnr_arr = np.array([x for x in psnr_list if not np.isnan(x)])
    ssim_arr = np.array([x for x in ssim_list if not np.isnan(x)])
    psnr_FA_arr = np.array([x for x in psnr_FA_list if not np.isnan(x)])
    ssim_FA_arr = np.array([x for x in ssim_FA_list if not np.isnan(x)])

    print("\n" + "="*60)
    print("数据集测试结果汇总 (均值 ± 标准差)")
    print("V4 Transformer编码器模型 + Mask增强")
    print("="*60)
    print(f"PSNR:     {psnr_arr.mean():.4f} ± {psnr_arr.std():.4f}")
    print(f"SSIM:     {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}")
    if len(psnr_FA_arr) > 0:
        print(f"FA PSNR:  {psnr_FA_arr.mean():.4f} ± {psnr_FA_arr.std():.4f}")
        print(f"FA SSIM:  {ssim_FA_arr.mean():.4f} ± {ssim_FA_arr.std():.4f}")
    print("="*60)

    # 保存结果到文件
    with open(os.path.join(result_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("数据集测试结果汇总 (均值 ± 标准差)\n")
        f.write("V4 Transformer编码器模型 + Mask增强\n")
        f.write("="*60 + "\n")
        f.write(f"PSNR:     {psnr_arr.mean():.4f} ± {psnr_arr.std():.4f}\n")
        f.write(f"SSIM:     {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}\n")
        if len(psnr_FA_arr) > 0:
            f.write(f"FA PSNR:  {psnr_FA_arr.mean():.4f} ± {psnr_FA_arr.std():.4f}\n")
            f.write(f"FA SSIM:  {ssim_FA_arr.mean():.4f} ± {ssim_FA_arr.std():.4f}\n")
        f.write("="*60 + "\n\n")
        f.write("各样本详细指标:\n")
        for i, sub_id in enumerate(processed_ids):
            f.write(f"{sub_id}: PSNR={psnr_list[i]:.4f}, SSIM={ssim_list[i]:.4f}")
            if i < len(psnr_FA_list) and not np.isnan(psnr_FA_list[i]):
                f.write(f", FA_PSNR={psnr_FA_list[i]:.4f}, FA_SSIM={ssim_FA_list[i]:.4f}")
            f.write("\n")
    
    print(f"指标已保存到 {os.path.join(result_dir, 'metrics.txt')}")

    return {
        'psnr': (psnr_arr.mean(), psnr_arr.std()),
        'ssim': (ssim_arr.mean(), ssim_arr.std()),
        'psnr_FA': (psnr_FA_arr.mean(), psnr_FA_arr.std()) if len(psnr_FA_arr) > 0 else (np.nan, np.nan),
        'ssim_FA': (ssim_FA_arr.mean(), ssim_FA_arr.std()) if len(ssim_FA_arr) > 0 else (np.nan, np.nan)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trans-TCDM on one dataset.")
    parser.add_argument("--config", required=True, help="Path to a dataset JSON config.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to save_dir/best.pth.")
    parser.add_argument("--subjects", nargs="+", help="Optional subject keys from the test HDF5 file.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--skip_ids", default="", help="Comma-separated subject keys to skip.")
    parser.add_argument("--force_resample", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config["gpus"] = [args.gpu]
    config["output_device"] = f"cuda:{args.gpu}"
    config["device"] = f"cuda:{args.gpu}"
    config["batch_size"] = 1
    config["checkpoint"] = args.checkpoint or os.path.join(config["save_dir"], "best.pth")
    config["subjects"] = args.subjects
    config["start_idx"] = args.start_idx
    config["skip_ids"] = [s.strip() for s in args.skip_ids.split(",") if s.strip()]
    config["force_resample"] = args.force_resample
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    evaluate(config)


if __name__ == "__main__":
    main()
