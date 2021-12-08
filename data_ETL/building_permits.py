import openpyxl as op
import helpers

master_tab = 'MSA Units'

excel_files = ['../src data/building permits/2019 building permits.xlsx',
               '../src data/building permits/2020 building permits.xlsx',
               '../src data/building permits/2016 units.xlsx',
               '../src data/building permits/2017 units.xlsx',
               '../src data/building permits/2018 units.xlsx']
db_file = '../housing_inventory.db'


def get_padded_months() -> list:
    padded_months = [''.join(['0', str(x)]) for x in range(1, 10)]
    padded_months.append('11')
    padded_months.append('12')
    return padded_months


def get_spreadsheet_info(t_file: str, t_padded_months: list) -> list:
    """
    Get the info out of the csv file and handle the year month breakout
    :param t_file:
    :return:
    """
    if '2020' in t_file:
        year = '2020'
    elif '2019' in t_file:
        year = '2019'
    elif '2018' in t_file:
        year = '2017'
    elif '2017' in t_file:
        year = '2017'
    elif '2016' in t_file:
        year = '2016'

    wb = op.load_workbook(t_file)
    results = []
    master_sheet = wb[master_tab]
    for cell_value in master_sheet.iter_rows(min_row=8, min_col=1, max_col=4, values_only=True):
        if cell_value[0] is None: break
        cbsa_code = cell_value[1]
        monthly_est = int(cell_value[3]) / 12  # we have annual rollups, let's estimate via mean
        month_entries = [[''.join([year, x]), cbsa_code, monthly_est] for x in t_padded_months]
        [results.append(month) for month in month_entries]
    return results


def fill_db(t_entries: list, db: dict) -> None:
    """
    Take the dictionary and fill the database
    """
    con = db['con']
    cur = db['cur']
    cur.executemany(
        'INSERT into building_permits(year_month, cbsa_code, total_units) VALUES (?, ?, ?)',
        t_entries)
    con.commit()
    return


def main(t_db_file = db_file):
    padded_months = get_padded_months()
    db_ref = helpers.create_connection(t_db_file)
    for spreadsheet in excel_files:
         permits = get_spreadsheet_info(spreadsheet, padded_months)
         fill_db(permits, db_ref)
    helpers.close_connection(db_ref['con'])
    return


if __name__ == '__main__':
    main()
