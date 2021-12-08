import sqlite3


def create_connection(db_file: str) -> dict:
    """
    Create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    db={}
    db['con'] = None
    db['cur'] = None
    try:
        db['con'] = sqlite3.connect(db_file)
        db['cur'] = db['con'].cursor()
    except sqlite3.Error as e:
        print(e)
    return db


def close_connection(db_conn: sqlite3.Connection) -> None:
    return db_conn.close()
