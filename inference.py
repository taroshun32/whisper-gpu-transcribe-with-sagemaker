import os
import logging
import uuid
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)


def model_fn(model_dir):
    logging.info('start model_fn')

    try:
        logging.info(f'Loading model from {model_dir}')
        model = WhisperModel(
            model_dir,
            device="cuda",
            compute_type="float16"
        )
    except Exception as e:
        logging.error(f"Error in model_fn: {str(e)}")
        raise

    logging.info('complete model_fn')
    return model


def input_fn(input_data, content_type):
    logging.info('start input_fn')

    try:
        if content_type == 'audio/mpeg':
            os.makedirs('/tmp', exist_ok=True)
            unique_filename = str(uuid.uuid4())
            local_path = f'/tmp/{unique_filename}.mp3'

            with open(local_path, 'wb') as f:
                f.write(input_data)
        else:
            raise ValueError(f"Illegal content type {content_type}. The only allowed content_type is audio/mpeg")
    except Exception as e:
        logging.error(f"Error in input_fn: {str(e)}")
        raise

    logging.info('complete input_fn')
    return local_path


def predict_fn(local_path, model):
    logging.info('start predict_fn')

    try:
        segments, info = model.transcribe(local_path, beam_size=5, vad_filter=True, without_timestamps=True)

        logging.info(f"Detected language '{info.language}' with probability {float(info.language_probability):.2f}")

        results = []
        for segment in segments:
            result = {
                "text": segment.text
            }
            results.append(result)
            logging.info(segment.text)
    except Exception as e:
        logging.error(f"Error predict_fn: {str(e)}")
        raise

    logging.info('complete predict_fn')
    return results, local_path


def output_fn(results, accept):
    logging.info('start output_fn')

    try:
        if accept == 'text/plain':
            transcription_text = "\n".join([segment['text'] for segment in results[0]])

            os.remove(results[1])

            return transcription_text, accept
        else:
            raise ValueError(f"Illegal accept {accept}. The only allowed content_type is text/plain")
    except Exception as e:
        logging.error(f"Error output_fn: {str(e)}")
        raise
