import speech_recognition as sr


class SpeechToText:
    """
    A class that helps to convert recorded audio into text

    """

    @staticmethod
    def audio_to_text(dur=3):
        """Convert Speech to Text.

        You can provide the duration for which the SpeechRecognition should run.

        Keyword arguments:
            dur -- (int) duration in seconds (default 3)

        Returns:
            A text converted from the user's speech.
        """

        recorder = sr.Recognizer()  # initializing the SpeechRecognition Class

        try:

            # using the device's default Microphone
            with sr.Microphone(device_index=0) as source:
                print("Listening...")
                # storing the recorded audio
                audio = recorder.record(source, duration=dur)

            text = recorder.recognize_google(
                audio)  # convert the audio into text
            print(f"You said: \"{text}\"")

            return text

        except Exception as err:

            print(f"Speech Recognition Error... Presuming that you said \"Hello\"")

            return "Hello"
