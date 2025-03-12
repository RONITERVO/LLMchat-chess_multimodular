# LLMchat-chess_multimodular
# LLM-Powered Multimodal Personal Assistant & Chess opponent

This project is a versatile personal assistant and chess opponent powered by Large Language Models (LLMs). It combines voice interaction, text chat, task management, multimedia integration, and a fully functional chess interface.
## You need: LLMchat.py and .json files; other .py files are optional.
## Key Features:

*   **Interactive Chat:** Engage in natural language conversations with a variety of LLMs through a text-based chat window.
*   **Task Management & Reminders:**  Set tasks and timers to boost your creativity and productivity.  Receive reminders from the LLM, complete with:
    *   **Contextually Relevant Music:**  Generated background music tailored to your task or reminder (via Mureka API).
    *   **Spoken Reminders:**  Reminders are spoken aloud using high-quality text-to-speech (OpenAI `tts-1`, all voice options supported).
    *   **Streaming LLM Responses:**  Hear the LLM's response as it's being generated, providing a more interactive and engaging experience.
*   **Chain-of-Thought Vocalization:**  For supported models, key parts of the LLM's reasoning process ("chain of thought") are spoken aloud, giving you insight into its decision-making, while the full text response is still generated.  This allows you to use both "thinking" and "non-thinking" LLMs.
*   **Multilingual Support:**  The core functionality currently supports English, Finnish, Spanish, and Swedish.  Instructions and structure are provided for adding more languages (though full support for new languages may require additional configuration).
*   **Multimedia Integration:**
    *   **Image Generation:**  The LLM can generate images based on its own reasoning or your explicit requests (via OpenAI DALL-E).
    *   **Image Input:**  You can send images to the LLM for analysis and discussion (supported by compatible models).
*   **Chess AI Opponent:**
    *   **Play Against Any LLM:**  Challenge a wide range of LLMs to a game of chess on a virtual chessboard.
    *   **Automatic Move Application:**  The LLM's moves are automatically applied to the board.
    *   **Flexible Input:**  Make your moves either by:
        *   **Voice Commands:**  Speak your moves naturally.
        *   **Manual Input:**  Drag and drop pieces on the chessboard.
* **Model Flexibility:** Choose between cloud-based LLMs through OpenRouter, or use your own locally hosted models with Ollama (optional, but provides greater privacy and control).

## Required APIs:

To use all features of this application, you will need API keys for the following services:

*   **OpenAI:**  Required for:
    *   `whisper-1`: Voice-to-text transcription.
    *   `tts-1`: High-quality text-to-speech.
    *   DALL-E: Image generation.
*   **Mureka (you will need useapi.net for access, follow instructions on the website):**  Required for LLM to generate contextual music for reminders and normal chat on request.
*   **OpenRouter:**  Provides access to a wide range of LLMs for chat and chess functionality.
*   **Ollama (Optional):**  Allows you to use locally hosted LLMs, providing an alternative to OpenRouter and enhanced privacy.  You will need to pull appropriate models (e.g., `qwq:32b`).

## Setup and Configuration:

1.  **Obtain API Keys:**  Create accounts and obtain API keys for the required services listed above.
2.  **Configuration File:**  Edit the `config.json` file to include your API keys and customize settings (e.g., default image directory, LLM preferences).
3.  **Install Dependencies:** Ensure you have the necessary Python libraries installed.  A `requirements.txt` file is highly recommended (see below).
4.  **Run the Application:** Execute the main Python script to start the assistant.

## Contributing:

Contributions are welcome!  Feel free to submit bug reports, feature requests, or pull requests.  Areas for potential improvement include:

*   Expanding language support.
*   Adding support for more LLMs and APIs.
*   Improving the user interface.
*   Refining the chess AI logic.

## License:

This project is licensed under the GNU General Public License v3.0 (or later).  See the LICENSE file for details.
