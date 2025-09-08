# Feel free to change the prompt and the image!
prompt="A bright yellow water taxi glides smoothly across the choppy waters, creating gentle ripples in its wake. The iconic Brooklyn Bridge looms majestically in the background, its intricate web of cables and towering stone arches standing out against the city skyline. The boat, bustling with passengers, offers a lively contrast to the serene, expansive sky dotted with fluffy clouds. As it cruises forward, the vibrant cityscape of New York unfolds, with towering skyscrapers and historic buildings lining the waterfront, capturing the dynamic essence of urban life."
img_path="examples/cog/img/boat.jpg"

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    cog_inference.py \
    --prompt "$prompt" \
    --image_path $img_path \
    --output_path "output-cog-uly2.mp4" \
    --ulysses_degree 2 \
    --use_sequence_parallel

