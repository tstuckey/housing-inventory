import openpyxl as op
import helpers
from datetime import datetime

master_tab = 'Sheet1'

src_file = '../src data/Prime Interest Rates.xlsx'
db_file = '../housing_inventory.db'


def get_info(t_file: str) -> list:
    """
    Get the info out of the csv file and handle the year month breakout
    :param t_file:
    :return:
    """
    wb = op.load_workbook(t_file)
    results=[]
    master_sheet = wb[master_tab]
    for cell_value in master_sheet.iter_rows(min_row=7, min_col=0, max_col=2, values_only=True):
        if cell_value[0] is None: break
        new_date = datetime.strptime(cell_value[0],'%Y-%m')
        new_date = datetime.strftime(new_date, '%Y%m')
        # only retain the records after June 2016 for import into the db
        if new_date > '201606':
            results.append([new_date, cell_value[1]])
    return results


def fill_db(t_entries: list, db: dict) -> None:
    """
    Take the dictionary and fill the database
    """
    con = db['con']
    cur = db['cur']
    cur.executemany(
        'INSERT into prime_rates(year_month, prime_rate) VALUES (?, ?)',
        t_entries)
    con.commit()
    return


def main(t_db_file = db_file):
    prime_rates= get_info(src_file)
    db_ref = helpers.create_connection(t_db_file)
    fill_db(prime_rates, db_ref)
    helpers.close_connection(db_ref['con'])
    return


if __name__ == '__main__':
    main()
