"""
Convert PyTorch .pth model to safetensors .st format
Usage: python convert_pth_to_st.py <input.pth> <output.st>
"""

import sys
import torch
from safetensors.torch import save_file
from tqdm import tqdm

def convert_pth_to_st(input_path: str, output_path: str):
    """
    Convert PyTorch .pth file to safetensors .st file
    
    Args:
        input_path: Path to input .pth file
        output_path: Path to output .st file
    """
    print(f"Loading PyTorch model from: {input_path}")
    
    # 加载 PyTorch 模型
    try:
        state_dict = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading .pth file: {e}")
        return False
    

    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    print(f"Found {len(state_dict)} tensors")
    
    converted_dict = {}
    for key, tensor in tqdm(state_dict.items(), desc="Processing tensors"):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.half()
        
        converted_dict[key] = tensor
    
    print(f"Saving safetensors model to: {output_path}")
    
    # 保存为 safetensors 格式
    try:
        save_file(converted_dict, output_path)
        print(f"✓ Successfully converted to: {output_path}")
        
        # 计算文件大小
        import os
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Input size:  {input_size:.2f} MB")
        print(f"Output size: {output_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error saving safetensors file: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_pth_to_st.py <input.pth> <output.st>")
        print("\nExample:")
        print("  python convert_pth_to_st.py model.pth model.st")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not input_path.endswith('.pth'):
        print("Warning: Input file should be .pth format")
    
    if not output_path.endswith('.st'):
        print("Warning: Output file should be .st format")
    
    success = convert_pth_to_st(input_path, output_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

