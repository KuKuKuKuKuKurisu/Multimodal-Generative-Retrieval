from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
save_path = "/home/student2020/shy/MIC-master/mmd_data"
output_path = "/home/student2020/shy/MIC-master/mmd_data/stage2_model.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)