# 🎵 Personalized Music Recommendation for Social Media Using AI

## 📌 Overview  
This project leverages **AI-driven mood detection** and **personalized music recommendations** to enhance Social Media posts. By analyzing the emotional tone of uploaded media and integrating user preferences via Spotify, the system provides tailored song suggestions to amplify engagement and creativity.

---

## 🛠️ Workflow  

### **Step 1: Image Input**
- Users upload an image or video to the system.
- **Prompt Example:** "Identify the mood of this image."

### **Step 2: Vision Model Analysis**
- A vision reasoning model (e.g., LLaVA) processes the uploaded media to classify its mood.
- **Output Example:**  
  - Mood: *Calm and Reflective*  
  - Features: **Low Valence (0.3)**, **Low Energy (0.2)**  

### **Step 3: Querying Database for the song**
- The system queries a database containing metadata for songs.  
- The database includes features like valence, energy, tempo and lyrics for each song.

### **Step 4: Generating Top Recommendations**
- Songs matching the specified feature range are retrieved from the database.  
- The system ranks songs based on similarity to the mood features and selects the top 10 recommendations.

### **Step 5: Spotify Integration**
- Users authenticate their Spotify account.  
- Recommendations are refined further by cross-referencing the user's listening history, playlists, and preferences.

### **Step 6: Final Output**
- The system presents a list of recommended songs tailored to the uploaded media's mood and user preferences.  
- Users can integrate these recommendations into their Instagram posts or stories seamlessly.

---

## 🚀 Key Features
1. Real-time mood detection using computer vision models.
2. Personalized music recommendations powered by Spotify API.
3. Interactive web application for media upload and music suggestion.
4. Seamless user authentication with Spotify for playlist integration.

---

This project combines cutting-edge AI techniques with creative media personalization, offering users a seamless way to enhance their Social Media content with perfectly matched music.
