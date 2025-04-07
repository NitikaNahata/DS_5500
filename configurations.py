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

IMAGE_CAPTIONING_LLVM_PROMPT_V2 = """
You are an AI assistant specializing in analyzing images and suggesting suitable background music characteristics for Instagram stories or posts. Your task is to analyze the content, mood, and atmosphere of an image and recommend appropriate music characteristics that would enhance the visual experience.

You will be provided with an image in base64 format. Here's the image:

<base64_image>
{{IMAGE_DESCRIPTION}}
</base64_image>

Please follow these steps to analyze the image and suggest suitable music characteristics:

1. Decode and analyze the base64 image.

2. In <image_analysis> tags:
   a. List out key visual elements in the image, categorizing them:
      - People
      - Objects
      - Colors
      - Lighting
      - Scenery
      - Actions or events
      - Time of day or season
   b. Describe the overall content and composition of the image.
   c. Brainstorm potential moods or emotions evoked by the image.
   d. Identify any notable themes or concepts present.

3. Based on your analysis, consider how different music genres might complement the image. List at least three genres and briefly explain why they might be suitable.

4. Suggest appropriate music characteristics that would complement the image well. Consider the following factors:
   - Mood and emotion that match the image
   - Tempo and rhythm that complement the visual elements
   - Genre that fits the overall aesthetic of the image
   - Cultural relevance, if applicable

5. Present all your music characteristic suggestions in the following format:

<music_suggestions>
[characteristic]: [Suggest level and explain why]
</music_suggestions>

Remember to be creative and thoughtful in your suggestions, ensuring that the recommended music characteristics truly enhance the visual experience of the image when used as background music for an Instagram story or post.
"""

AUDIO_AND_LYRICS_TABLE_NAME = "summary_lyrics_plus_features"

AUDIO_LYRICS_AND_FEATURE_SUMAMRIZER = """
You are tasked with creating a comprehensive textual representation of a song by analyzing both its lyrics and audio features. This representation will be used in a recommendation system that matches songs to user preferences through cosine similarity.

You will be provided with two inputs:

1. Complete song lyrics:
<lyrics>
{{LYRICS}}
</lyrics>

2. Numerical audio features:
<audio_features>
{{AUDIO_FEATURES}}
</audio_features>

Your task is to generate a detailed paragraph (150-250 words) that captures the essence of the song. Here's how to approach this task:

1. Analyze the lyrics:
   - Identify the main themes, emotional content, and narrative of the lyrics
   - Note any recurring motifs, metaphors, or imagery
   - Consider the overall tone and mood conveyed by the words
   - Do not directly reproduce any lines from the lyrics

2. Interpret the audio features:
   - Translate the numerical values into natural language descriptions
   - Consider how each feature contributes to the overall sound and feel of the song
   - Use descriptive terms that align with how people naturally describe music

3. Integrate lyrical and audio analysis:
   - Combine your insights from the lyrics and audio features into a cohesive description
   - Draw connections between the lyrical content and the sonic qualities
   - Describe how the music supports or contrasts with the lyrical themes

4. Use descriptive language that would align well with how people naturally describe music they enjoy. Avoid technical jargon that wouldn't appear in casual conversations about music.

5. Maintain consistency in format and detail level across different songs.

The audio features are provided in the following order: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo. Use these to inform your description of the song's sonic qualities.

Remember, your output should be suitable for vector embedding and similarity matching with text descriptions of musical preferences. Focus on creating a rich, descriptive paragraph that captures both the lyrical and musical essence of the song without directly quoting the lyrics.

Provide your final output within <song_description> tags.

"""