from flask import Flask, request, Response
from transformers import pipeline
import waitress
import logging
import click
import json
import sys

model_names = [
    'aisquared/dlite-v2-124m',
    'aisquared/dlite-v2-355m',
    'aisquared/dlite-v2-774m',
    'aisquared/dlite-v2-1_5b'
]
simplified_names = [n.split('/')[1] for n in model_names]

logger = logging.getLogger(__name__)

logging.basicConfig(
    format = '%(levelname)s | %(asctime)s | %(message)s',
    datefmt = '%Y-%m-%dT%H:%M:%SZ',
    stream = sys.stdout,
    level = logging.INFO
    )

def deploy_model(
        host = '0.0.0.0',
        port = 2244
):
    
    app = Flask(__name__)

    logger.info('loading models')

    models = {}
    for idx, model in enumerate(model_names):
        
        logger.info(f'attempting to load {model}')
        
        try:
            models[simplified_names[idx]] = pipeline(model = model, trust_remote_code = True, device_map = 'auto')
            logger.info(f'model {model} successfully loaded')
        
        except Exception as e:
            logger.exception(f'Error while loading model {model}: {e}')

    @app.route('/predict', methods = ['POST'])
    def predict():
        
        logger.info('request received')

        # Retrieve the json
        try:
            data = request.get_json()
            logger.info('data successfully parsed from request')
        except Exception as e:
            logger.exception(f'Error while parsing data from request: {e}')
            return Response(
                'No parsable payload',
                400
            )
        
        # Retrieve the prompt
        try:
            prompt = data['prompt']
            logger.info('prompt parsed from request data')
        except Exception as e:
            logger.exception(f'Error while retrieving prompt: {e}')
            return Response(
                'No prompt detected',
                400
            )
        
        # Retrieve the model
        try:
            model = models[data['model']]
        except Exception as e:
            logger.exception(f'Error while retrieving model: {e}')
            return Response(
                f'model either not specified or not one of {simplified_names}',
                400
            )
        
        try:
            model_response = model(prompt)
            logger.info('model response to prompt successfully calculated')
            model_response = {
                'choices' : [{'text' : model_response}]
            }
            logger.info('model response successfully formatted, returning response')
            return json.dumps(model_response)
        
        except Exception as e:
            logger.exception(f'Error while retrieving model response: {e}')
            return Response(
                'Error in performing prediction',
                500
            )

    # Serve the application
    logger.info('Serving models')
    waitress.serve(
        app,
        host = host,
        port = port
    )

@click.command()
def main():
    deploy_model()

if __name__ == '__main__':
    main()
