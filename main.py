from src.tests import n_dimentional_cech_complex_test, enumerate_simplexes_ck_test, enumerate_simplexes_ckl_test

if __name__ == "__main__":
    # It does all the basic tests proposed at 2.2
    n_dimentional_cech_complex_test()
    # It does task 2 on the required example

    print("Enumerate Ck Simplex (k=4)")
    enumerate_simplexes_ck_test()
    print()

    print("Enumerate Ckl Simplex (k=4, l=4)")
    enumerate_simplexes_ckl_test()
    print()
