FINAL_VISION_MODEL = "minicpm-v:8b"

FINAL_REASONING_MODEL = "deepseek-r1:14b"

IMAGE_CAPTIONING_LLVM_PROMPT_old = """
Analyze the image and provide a detailed response in keywords only. Follow the structure below:
 
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
"""

LLM_REASONING_MODEL = "deepseek-r1:14b"

LLM_REASONING_MODEL_V2 = "qwq:32b"

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

IMAGE_CAPTIONING_LLVM_PROMPT_V3 = """

You are a social media influencer tasked with describing an image in a way that can be used by a musician to suggest the kind of music that would complement the image. Your description should be so vivid and detailed that it captures the essence of the image, its mood, and its potential musical atmosphere.

Here is the base64 encoded image you need to analyze:
<image_base64>
{{IMAGE_BASE64}}
</image_base64>

Carefully examine the image and consider the following aspects:
1. The overall scene or setting
2. The main subjects or focal points
3. Colors, lighting, and atmosphere
4. Emotions or moods evoked by the image
5. Any actions or activities depicted
6. Time of day or season, if apparent
7. Style or aesthetic of the image (e.g., vintage, modern, minimalist)

Provide your description of the image inside <image_description> tags. Your description should:
- Be detailed and vivid, painting a clear picture with words
- Capture the mood and atmosphere of the image
- Highlight elements that could inspire musical choices
- Use language that a social media influencer would use, making the description engaging and relatable
- Be around 150-200 words in length

Remember, you are describing this image as a social media influencer would. Your description should be engaging, evocative, and designed to resonate with your followers. Think about how the image makes you feel and what kind of story it tells.

Your description will be used to suggest music that complements the image, so be sure to include details that could inspire musical choices. This might include the overall mood, the energy level, any cultural references, or specific elements that could correspond to musical genres or styles.

Do not mention anything about music recommendations directly in your description. Focus solely on describing the image in a way that a musician could later use to make appropriate music suggestions.

Begin your response now:

<image_description>
"""

USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT = """
You are a reasoning model tasked with analyzing an image and determining the most suitable music for it in the context of a social media post. You will be given two inputs: a description from the user about the kind of music they want for their image, and a description of the image from a vision model acting as a social media influencer.

Here is the user's description of the desired music:
<user_description>
{user_description}
</user_description>

Now, here is the vision model's description of the image:
<vision_description>
{vision_description}
</vision_description>

Analyze both inputs carefully. Consider the mood, atmosphere, and context conveyed by the image description. Think about how different types of music might complement or contrast with the image.

Based on your analysis, determine what kind of music would be ideal for this image in a social media post context. Consider genres, tempos, moods, and specific musical elements that would enhance the image's impact.

Next, evaluate whether the user's suggested music aligns well with the image. If it does, explain why. If it doesn't, explain why not and suggest how it could be improved.

Provide your response in the following format:
<analysis>
1. Brief summary of the image based on the vision model's description
2. Your reasoning on the ideal music for this image
3. Evaluation of the user's music suggestion
4. If applicable, suggestions for improvement or alternatives
</analysis>

<music_recommendation>
Concise statement of the recommended music style or genre
</music_recommendation>

Remember, your final output should only include the <analysis> and <music_recommendation> sections. Do not include any additional commentary or explanations outside these tags.

"""

USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT_V2 = """
You are an AI assistant specialized in analyzing visual content and recommending suitable music for social media posts. Your task is to determine the most appropriate music for an image based on two inputs: a description of the desired music from the user and a description of the image from a vision model.

Here are the inputs you'll be working with:

1. User's description of desired music:
<user_music_description>
{user_music_description}
</user_music_description>

2. Vision model's description of the image:
<image_description>
{image_description}
</image_description>

Your goal is to analyze both inputs and provide a recommendation for the most suitable music for this image in the context of a social media post. Follow these steps:

1. Carefully read and analyze both the user's music description and the image description.
2. Consider the mood, atmosphere, and context conveyed by the image description.
3. Think about how different types of music might complement or contrast with the image.
4. Determine what kind of music would be ideal for this image in a social media post context. Consider genres, tempos, moods, and specific musical elements that would enhance the image's impact.
5. Evaluate whether the user's suggested music aligns well with the image.
6. If the user's description and the image description significantly differ, prioritize the user's preferences by 85% when making your recommendation.

Before providing your final recommendation, wrap your analysis inside <music_image_analysis> tags in your thinking block. In your analysis, include:

1. A brief summary of the image based on the vision model's description
2. Your reasoning on the ideal music for this image, considering:
   a. Specific musical elements (tempo, instruments, vocals)
   b. How these elements complement the image
   c. The target audience for the social media post
3. An evaluation of the user's music suggestion
4. If applicable, suggestions for improvement or alternatives

After your analysis, provide your final music recommendation in <music_recommendation> tags.

Example output structure (do not copy the content, only the structure):

<music_image_analysis>
1. Image summary: [Brief description of the image]
2. Ideal music reasoning:
   a. Musical elements: [Specific elements that would suit the image]
   b. Complementary aspects: [How these elements enhance the image]
   c. Target audience considerations: [How the music appeals to the intended audience]
3. Evaluation of user's suggestion: [Assessment of how well the user's music idea fits]
4. Improvement suggestions: [If needed, ideas for better music choices]
</music_image_analysis>

<music_recommendation>
[Concise statement of the recommended music style or genre]
</music_recommendation>

Remember to consider both the image content and the user's preferences, especially if they differ significantly. Your goal is to provide a thoughtful, well-reasoned recommendation that enhances the social media post's impact.

Your final output should consist only of the music recommendation in <music_recommendation> tags and should not duplicate or rehash any of the work you did in the thinking block.
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