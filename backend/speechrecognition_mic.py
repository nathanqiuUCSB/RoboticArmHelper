import speech_recognition as sr

def speech_to_text(timeout=5, phrase_time_limit=10):
    """
    Records audio from the microphone and returns transcribed text.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
        except sr.WaitTimeoutError:
            return "No speech detected."

    try:
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        return "Could not understand audio."

    except sr.RequestError as e:
        return f"Speech recognition error: {e}"
if __name__ == "__main__":
    result = speech_to_text()
    print("📝 Transcribed text:", result)