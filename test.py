from elevenlabs import ElevenLabs

client = ElevenLabs(api_key="sk_abc73271e7e087fb3cecab7d5b132849d2585745c2a935c5")

voice_id = "6JsmTroalVewG1gA6Jmw"  # Rachel
model_id = "eleven_multilingual_v2"

text = "Hello! I'm your AI assistant from UBIK Solutions. How can I help you today?"

# Generate the streaming audio
audio_stream = client.text_to_speech.convert(
    voice_id=voice_id,
    model_id=model_id,
    text=text
)

# Save the audio to file
with open("test_outpu1t.mp3", "wb") as f:
    for chunk in audio_stream:
        f.write(chunk)

print("✅ Audio generated and saved as test_output.mp3")
