huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers

cd ..
python svg/models/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder

cd ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2