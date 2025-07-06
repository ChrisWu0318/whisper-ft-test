import argparse
import functools
import platform

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from utils.utils import print_arguments, add_arguments

# Argument parsing
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="dataset/test.wav", help="预测的音频路径")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune/", help="模型路径或 Huggingface 模型名")
add_arg("use_gpu",     type=bool, default=True,      help="是否使用 GPU")
add_arg("language",    type=str,  default="chinese", help="语言（小写或全称），为空则为多语言")
add_arg("num_beams",   type=int,  default=1,         help="beam search 数量")
add_arg("batch_size",  type=int,  default=16,        help="预测 batch size")
add_arg("use_compile", type=bool, default=False,     help="是否启用 torch.compile")
add_arg("task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="任务类型")
add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型（可选）")
add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用 FlashAttention2 加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用 BetterTransformer 加速")
args = parser.parse_args()
print_arguments(args)

# Device and dtype
device = "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32

# Processor
processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)

# 判断是否是 Whisper 模型（不支持 FlashAttention）
is_whisper = "whisper" in args.model_path.lower()

# 模型加载参数处理
model_kwargs = {
    "torch_dtype": torch_dtype,
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
    "local_files_only": args.local_files_only
}
if args.use_flash_attention_2 and not is_whisper:
    model_kwargs["use_flash_attention_2"] = True

# 加载模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_path, **model_kwargs)
model.generation_config.forced_decoder_ids = None

# BetterTransformer 加速
if args.use_bettertransformer and not args.use_flash_attention_2:
    model = model.to_bettertransformer()

# PyTorch 2.0 编译器
if args.use_compile and torch.__version__ >= "2" and platform.system().lower() != 'windows':
    model = torch.compile(model)

model.to(device)

# 助手模型加载（可选）
generate_kwargs_pipeline = {"max_new_tokens": 128}
if args.assistant_model_path:
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model_path, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True
    )
    assistant_model.to(device)
    generate_kwargs_pipeline["assistant_model"] = assistant_model

# 推理 pipeline
infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      chunk_length_s=30,
                      batch_size=args.batch_size,
                      torch_dtype=torch_dtype,
                      generate_kwargs=generate_kwargs_pipeline,
                      device=device)

# 推理参数
generate_kwargs = {"task": args.task, "num_beams": args.num_beams}
if args.language:
    generate_kwargs["language"] = args.language.lower()

# 执行推理
result = infer_pipe(args.audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

# 输出每段
print("\n===== 分段识别结果 =====")
for chunk in result["chunks"]:
    print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")

# 输出全文
print("\n===== 整体识别文本 =====")
print(result["text"])

