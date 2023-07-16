import dotenv # pip install python-dotenv
import os

class ServiceConfiguration:
    # Load HOST and PORT from .env file
    def load_config():
        dotenv.load_dotenv()
        HOST = os.getenv('HOST')
        PORT = os.getenv('PORT')

        return HOST, PORT
    

