import os
import shutil
from pathlib import Path

def copy_adapter_file():
    # 源文件路径
    source_path = "Heter-gpt2-dolly-homo/4/local_output_0/adapter.bin"
    
    # 目标文件路径（在Heter-gpt2-dolly-homo文件夹下）
    target_dir = "Heter-gpt2-dolly-homo"
    target_path = os.path.join(target_dir, "adapter.bin")  # 保持相同文件名
    
    try:
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            print(f"错误：源文件 '{source_path}' 不存在！")
            return False
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        print(f"文件已成功复制：")
        print(f"  从: {os.path.abspath(source_path)}")
        print(f"  到: {os.path.abspath(target_path)}")
        
        # 验证复制是否成功
        if os.path.exists(target_path):
            print(f"文件复制成功！新文件大小为: {os.path.getsize(target_path)} 字节")
            return True
        else:
            print("错误：文件复制后目标文件不存在！")
            return False
            
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        return False
    except PermissionError as e:
        print(f"权限错误: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

# 使用pathlib的替代方案（更现代的方法）
def copy_adapter_file_pathlib():
    # 源文件路径
    source_path = Path("./Heter-gpt2-dolly-homo/4/local_output_0/adapter.bin")
    
    # 目标文件路径
    target_dir = Path("./Heter-gpt2-dolly-homo")
    target_path = target_dir / "adapter.bin"
    
    try:
        # 确保目标目录存在
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查源文件是否存在
        if not source_path.exists():
            print(f"错误：源文件 '{source_path}' 不存在！")
            return False
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        print(f"文件已成功复制：")
        print(f"  从: {source_path.absolute()}")
        print(f"  到: {target_path.absolute()}")
        
        # 验证复制是否成功
        if target_path.exists():
            print(f"文件复制成功！新文件大小为: {target_path.stat().st_size} 字节")
            return True
        else:
            print("错误：文件复制后目标文件不存在！")
            return False
            
    except Exception as e:
        print(f"发生错误: {e}")
        return False

if __name__ == "__main__":
    # 使用方法1（传统os方法）
    print("使用方法1复制文件...")
    success = copy_adapter_file()
    
    if not success:
        print("\n使用方法2复制文件...")
        success = copy_adapter_file_pathlib()
    
    if success:
        print("\n操作完成！")
    else:
        print("\n操作失败，请检查路径和文件权限。")
