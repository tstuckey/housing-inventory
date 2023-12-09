import housing_inventory as hi
import mortage_rates as mr
import revolving_credit as rc
import building_permits as bp
import prime_rate as pr
import build_aggregate_table as agg

db_file = '../housing_inventory.db'


def populate_db(t_db_file: str) -> None:
    hi.main(t_db_file)
    mr.main(t_db_file)
    rc.main(t_db_file)
    bp.main(t_db_file)
    pr.main(t_db_file)
    agg.main(t_db_file)


if __name__ == '__main__':
    populate_db(db_file)
