// static/js/main.js
document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("book-file");
  const welcomeMessage = document.querySelector(".welcome-message");
  const processingMessage = document.querySelector(".processing-message");
  const audioContainer = document.querySelector(".audio-container");
  const audioPlayer = document.getElementById("audioPlayer");
  const generateButton = document.getElementById("generateButton");
  const downloadButton = document.getElementById("downloadButton");
  const chatMessages = document.getElementById("chat-messages");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");

  // Global variables
  let currentBook = null;
  let currentAudioUrl = null;

  // Handle file upload
  uploadForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const file = fileInput.files[0];

    if (!file) {
      alert("Please select a file to upload");
      return;
    }

    formData.append("file", file);

    try {
      // Show processing message
      welcomeMessage.style.display = "none";
      processingMessage.style.display = "block";
      audioContainer.style.display = "none";

      // Upload the file
      const uploadResponse = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const uploadData = await uploadResponse.json();

      if (uploadData.success) {
        currentBook = uploadData.book;

        // Hide processing message, show audio container with generate button
        processingMessage.style.display = "none";
        audioContainer.style.display = "block";
        generateButton.disabled = false;

        // Set up generate button
        generateButton.onclick = generatePodcast;
      } else {
        alert(uploadData.error || "Failed to upload file");
        resetUI();
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred. Please try again.");
      resetUI();
    }
    handleAudioErrors();
  });

  async function generatePodcast() {
    try {
        // Show processing state
        generateButton.disabled = true;
        generateButton.textContent = '🔄 Generating...';
        audioPlayer.style.display = 'none';
        downloadButton.style.display = 'none';
        
        // Generate podcast
        const podcastResponse = await fetch(`/generate_podcast/${encodeURIComponent(currentBook.title)}`);
        
        if (!podcastResponse.ok) {
            throw new Error(`Server error: ${podcastResponse.status}`);
        }
        
        const podcastData = await podcastResponse.json();
        
        if (podcastData.success) {
            // Create audio source (changed to audio/mpeg for MP3 format)
            const audioSrc = `data:audio/mpeg;base64,${podcastData.podcast.audio_base64}`;
            audioPlayer.src = audioSrc;
            currentAudioUrl = audioSrc;
            
            // Show audio player and download button
            audioPlayer.style.display = 'block';
            downloadButton.style.display = 'inline-block';
            
            // Set up error handling for audio playback
            audioPlayer.onerror = function() {
                console.error("Audio playback error:", this.error);
                alert("Error playing audio. Please try generating again.");
            };
            
            // Reset generate button
            generateButton.textContent = '🎙️ Regenerate Podcast';
            generateButton.disabled = false;
            
            // Set up download button
            downloadButton.onclick = function() {
                const link = document.createElement('a');
                link.href = audioSrc;
                link.download = `${currentBook.title.replace(/[^a-z0-9]/gi, '_')}_podcast.mp3`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
            
            // Add chat message about successful generation
            addChatMessage({
                text: `I've generated a podcast for "${currentBook.title}"! Listen below and ask me questions about the book!`,
                isAI: true
            });
            
        } else {
            throw new Error(podcastData.error || 'Failed to generate podcast');
        }
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
        generateButton.textContent = '🎙️ Generate Podcast';
        generateButton.disabled = false;
    }
}

  // // Generate podcast function
  // async function generatePodcast() {
  //   try {
  //     // Show processing state
  //     generateButton.disabled = true;
  //     generateButton.textContent = '🔄 Generating...';
  //     audioPlayer.style.display = 'none';
  //     downloadButton.style.display = 'none';
      
  //     // Generate podcast
  //     const podcastResponse = await fetch(`/generate_podcast/${encodeURIComponent(currentBook.title)}`);
  //     const podcastData = await podcastResponse.json();
      
  //     if (podcastData.success) {
  //       // Show audio player and download button
  //       audioPlayer.style.display = 'block';
  //       downloadButton.style.display = 'inline-block';
        
  //       // Set up audio player
  //       const audioSrc = `data:audio/mpeg;base64,${podcastData.podcast.audio_base64}`;
  //       audioPlayer.src = audioSrc;
  //       currentAudioUrl = audioSrc;
        
  //       // Reset generate button
  //       generateButton.textContent = '🎙️ Regenerate Podcast';
  //       generateButton.disabled = false;
        
  //       // Set up download button
  //       downloadButton.onclick = function() {
  //         const link = document.createElement('a');
  //         link.href = audioSrc;
  //         link.download = `${currentBook.title}_podcast.mp3`;
  //         document.body.appendChild(link);
  //         link.click();
  //         document.body.removeChild(link);
  //       };
        
  //       // Add chat message about successful generation
  //       addChatMessage({
  //         text: `I've generated a podcast for "${currentBook.title}"! Feel free to ask questions about the book!`,
  //         isAI: true
  //       });
        
  //       // Start playing the podcast
  //       audioPlayer.play();
  //     } else {
  //       alert(podcastData.error || 'Failed to generate podcast. Please try again.');
  //       generateButton.textContent = '🎙️ Generate Podcast';
  //       generateButton.disabled = false;
  //     }
  //   } catch (error) {
  //     console.error('Error:', error);
  //     alert('An error occurred while generating the podcast. Please try again.');
  //     generateButton.textContent = '🎙️ Generate Podcast';
  //     generateButton.disabled = false;
  //   }
  // }

  function handleAudioErrors() {
    audioPlayer.addEventListener('error', function() {
        console.error("Audio Error:", this.error);
        addChatMessage({
            text: "⚠️ Couldn't play the audio. Try regenerating the podcast.",
            isSystem: true
        });
    });
    
    audioPlayer.addEventListener('canplay', function() {
        console.log("Audio is ready to play");
    });
}

  // Reset UI to initial state
  function resetUI() {
    welcomeMessage.style.display = "block";
    processingMessage.style.display = "none";
    audioContainer.style.display = "none";
    audioPlayer.src = "";
    generateButton.textContent = "🎙️ Generate Podcast";
    generateButton.disabled = false;
    currentBook = null;
    currentAudioUrl = null;
    messageInput.disabled = true;
    sendBtn.disabled = true;
    chatMessages.innerHTML = "";
  }

  // Chat functionality
  sendBtn.addEventListener("click", sendMessage);

  messageInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

  function sendMessage() {
    const message = messageInput.value.trim();

    if (!message) return;

    if (!currentBook) {
      alert("Please select a book first");
      return;
    }

    // Add user message to chat
    addChatMessage({
      text: message,
      isUser: true,
    });

    // Clear input
    messageInput.value = "";

    // Send message to server
    fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        book_title: currentBook.title,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success && data.message) {
          // Add AI response to chat
          addChatMessage({
            text: data.message.text,
            isAI: true,
          });
        } else {
          console.error("Error:", data.error);

          // Add error message to chat
          addChatMessage({
            text: data.error || "Sorry, I had trouble understanding that. Could you try asking in a different way?",
            isSystem: true,
          });
        }
      })
      .catch((error) => {
        console.error("Error:", error);

        // Add error message to chat
        addChatMessage({
          text: "Sorry, there was a technical problem. Please try again.",
          isSystem: true,
        });
      });
  }

  function addChatMessage(message) {
    const messageElement = document.createElement("div");
    messageElement.className = `chat-message ${
        message.isUser ? 'user' : 
        message.isAI ? 'ai' : 
        'system'
    }`;
    
    const icon = message.isUser ? '👤' : 
                message.isAI ? '🤖' : 
                '⚠️';
    
    messageElement.innerHTML = `<p>${icon} ${message.text}</p>`;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
    

//   function addChatMessage(message) {
//     const messageElement = document.createElement("div");
//     messageElement.className = `chat-message ${message.isUser ? 'user' : message.isAI ? 'ai' : 'system'}`;
    
//     if (message.isUser) {
//       messageElement.innerHTML = `<p>${message.text}</p>`;
//     } else if (message.isAI) {
//       messageElement.innerHTML = `<p>🤖 ${message.text}</p>`;
//     } else {
//       messageElement.innerHTML = `<p>ℹ️ ${message.text}</p>`;
//     }
    
//     chatMessages.appendChild(messageElement);
//     chatMessages.scrollTop = chatMessages.scrollHeight;
//   }
 });
