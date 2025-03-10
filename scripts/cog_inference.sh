prompt="A cat with wide, curious eyes peers intently into the camera, its gaze steady and unblinking. The monochromatic tones highlight the intricate patterns of its fur and the delicate curve of its whiskers. As the video progresses, the cat's ears twitch slightly, picking up subtle sounds in the environment. Its nose quivers with each breath, adding a sense of gentle motion to the scene. The soft texture of the carpet above frames its face, creating a cozy, intimate atmosphere. The cat remains still, its expression a mix of curiosity and calm, drawing the viewer into its serene world."
img_path="examples/cog/img/cat.jpg"

python3 cog_inference.py --prompt "$prompt" --image_path $img_path --output_path "output.mp4"
