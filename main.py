from src.tests import n_dimentional_cech_complex_test, enumerate_simplexes_ck_test, enumerate_simplexes_ckl_test, simplex_in_alpha_complex_test

if __name__ == "__main__":
    # Task 1
    # It does all the basic tests proposed at 2.2
    n_dimentional_cech_complex_test()
    print("Task 1) All tests passed")
    print()

    # Task 2
    # It does task 2 on the required example
    print("Task 2) Enumerate Ck Simplex (k=4)")
    enumerate_simplexes_ck_test()
    print()

    # Task 3
    print("Task 3) Enumerate Ckl Simplex (k=4, l=4)")
    enumerate_simplexes_ckl_test()
    print()

    # Task 4
    print("Task 4) Test if simplex is in the alpha-complex")
    simplex_in_alpha_complex_test()
    print()
