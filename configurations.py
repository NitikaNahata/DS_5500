## This file will have all the constants, for example:
# OPENAI_VISION_MODEL = gpt-4o 


IMAGE_CAPTIONING_LLVM_PROMPT = """Analyze the image and provide a detailed response in keywords only. Follow the structure below:

Example 1:
Image Description: A calm forest scene with tall trees, soft sunlight filtering through leaves, and a small stream.
1. Main Theme(s): Serenity and Tranquility, Intimacy and Connection.
2. Key Elements: Green trees, soft lighting, flowing water.
3. Emotional Tone: Peaceful, refreshing.
4. Suggested Music Traits: High acousticness, low tempo, moderate valence.

Example 2:
Image Description: A bustling city street at night with bright neon lights and people walking.
1. Main Theme(s): Energy and Vitality, Mystery and Intrigue.
2. Key Elements: Neon lights, crowded streets, night sky.
3. Emotional Tone: Exciting, vibrant.
4. Suggested Music Traits: High energy, high tempo, low acousticness.

Now analyze this image (attached) in the same format:
Image Description: [Automatically extracted by model].
1. Main Theme(s):
2. Key Elements:
3. Emotional Tone:
4. Suggested Music Traits:"""