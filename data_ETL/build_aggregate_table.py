import helpers
import sqlite3

db_file = '../housing_inventory.db'


def build_main_table(db_ref: dict) -> None:
    con = db_ref['con']
    cur = db_ref['cur']
    query = """
    SELECT  bp.total_units as 'housing_permits', hi.total_listing_count,
        mr.mortgage_rate, pr.prime_rate, rc.credit,
       bp.cbsa_code, bp.year_month
    FROM building_permits as bp
    INNER JOIN housing_inventory hi
        on bp.year_month = hi.year_month and bp.cbsa_code = hi.cbsa_code
    INNER JOIN mortgage_rates mr
        on bp.year_month = mr.year_month
    INNER JOIN prime_rates pr
        on bp.year_month = pr.year_month
    INNER JOIN revolving_credit rc
        on bp.year_month = rc.year_month
    """
    result = list(con.execute(query))
    cur.executemany(
        'INSERT into overall(permits, mortgage_rate, prime_rate, credit, total_listing_count, cbsa_code, year_month) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        result)
    con.commit()
    return


def main(t_db_file= db_file):
    db_ref = helpers.create_connection(t_db_file)
    build_main_table(db_ref)
    helpers.close_connection(db_ref['con'])
    return


if __name__ == '__main__':
    main()
