import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def register_custom_modules():
	"""注册 SDA-STD YOLO11 自定义模块，确保自定义 YAML 可被 Ultralytics 解析。"""
	import sys
	# 复用 train.py 中的完整注册逻辑（含 parse_model 动态 patch）
	scripts_dir = str(Path(__file__).resolve().parent)
	if scripts_dir not in sys.path:
		sys.path.insert(0, scripts_dir)
	from train import register_custom_modules as _full_register
	_full_register(verbose=False)


def print_model_overview(title: str, model_source):
	"""加载模型并打印结构摘要。"""
	print("\n" + "=" * 100)
	print(title)
	print(f"模型来源: {model_source}")
	print("=" * 100)

	model = YOLO(str(model_source), task="detect")
	model.info(verbose=True)

	params = sum(parameter.numel() for parameter in model.model.parameters())
	gradients = sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad)

	print("-" * 100)
	print(f"参数总量: {params:,}")
	print(f"可训练参数: {gradients:,}")
	print("-" * 100)


def main():
	register_custom_modules()

	custom_model_path = PROJECT_ROOT / "models" / "MRA-STD YOLO.yaml"
	base_model_source = "yolo11s.yaml"

	print_model_overview("自定义 SDA-STD YOLO11 模型架构", custom_model_path)
	print_model_overview("标准 YOLO11s 模型架构（使用 Ultralytics 内置 yolo11s.yaml）", base_model_source)


if __name__ == "__main__":
	main()