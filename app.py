"""
This script runs the app application using a development server.
"""

from os import environ
from app import create_app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '443'))
    except ValueError:
        PORT = 443
    app = create_app()
    app.run(HOST, PORT, debug=True)
