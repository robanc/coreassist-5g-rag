import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection

from config import DATABASE_URL


def get_connection() -> Connection:
    connection = psycopg.connect(DATABASE_URL)
    register_vector(connection)

    return connection