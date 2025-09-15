import snowflake.connector

def get_connection(user, account, password, warehouse, database, schema, role):
    """
    Establishes and returns a Snowflake connection & cursor.
    """
    conn = snowflake.connector.connect(
        user=user,
        account=account,
        password=password,
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role
    )
    return conn, conn.cursor()


def run_query(cur, query):
    """
    Executes a SQL query and returns the results.
    """
    cur.execute(query)
    return cur.fetchall()


def close_connection(cur, conn):
    """
    Safely closes cursor and connection.
    """
    cur.close()
    conn.close()
