from app.services.whatsapp import detect_message_type, get_media_data


def test_embedded_openwa_voice_is_detected_and_extracted():
    payload = {
        "type": "voice",
        "media": {
            "mimetype": "audio/ogg; codecs=opus",
            "data": "T2dnUw==",
        },
    }

    assert detect_message_type(payload) == "voice"
    assert get_media_data(payload) == "T2dnUw=="


def test_audio_mime_type_is_detected_even_without_voice_type():
    payload = {"type": "unknown", "media": {"mimetype": "audio/ogg"}}

    assert detect_message_type(payload) == "voice"
