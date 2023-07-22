import numpy as np
from agreemat import kappa_matrix


#################################################################
#  DEFINE FUNCTIONS USED FOR UNIT TESTING
#################################################################
def test_1():
    """Test simple case."""
    X = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0, 0, 0, 0]]).transpose()
    K, A = kappa_matrix(X, return_agreement=True)
    K_ = np.array([[1, 1, 0, -1],
                [1, 1, 0, -1],
                [0, 0, 1, 0],
                [-1, -1, 0, 1]])
    A_ = np.array([[1, 1, .5, 0],
                [1, 1, .5, 0],
                [.5, .5, 1, .5],
                [0, 0, .5, 1]])
    assert (K==K_).all()
    assert (A==A_).all()


def test_2():
    """Test X with nan for all columns (items) in final row (person)"""
    X = np.array([[0., 0., 0., 0., 1., 1., 1., 1., np.nan],
                [0., 0., 0., 0., 1., 1., 1., 1., np.nan],
                [0., 1., 0., 1., 0., 1., 0., 1., np.nan],
                [1., 1., 1., 1., 0., 0., 0., 0., np.nan]]).transpose()
    K, A = kappa_matrix(X, return_agreement=True)
    K_ = np.array([[1, 1, 0, -1],
                [1, 1, 0, -1],
                [0, 0, 1, 0],
                [-1, -1, 0, 1]])
    A_ = np.array([[1, 1, .5, 0],
                [1, 1, .5, 0],
                [.5, .5, 1, .5],
                [0, 0, .5, 1]])
    assert (K==K_).all()
    assert (A==A_).all()


def test_3():
    """Test X with nan in different rows (people) for different columns (items)"""
    r1 = np.tile([0, 0, 0, 0, 1, 1, 1, 1], (1, 3))
    r2 = np.c_[np.array([[0, 0, 0, 0, 1, 1, 1, 1]]), np.full((1, 8*2), np.nan)]
    r3 = np.c_[np.full((1, 8*1), np.nan), np.array([[0, 1, 0, 1, 0, 1, 0, 1]]), np.full((1, 8*1), np.nan)]
    r4 = np.c_[np.full((1, 8*2), np.nan), np.array([[1, 1, 1, 1, 0, 0, 0, 0]])]
    X = np.r_[r1, r2, r3, r4].transpose()
    K, A = kappa_matrix(X, return_agreement=True)
    assert (K[0,:] == np.array([1, 1, 0, -1])).all()
    assert (K[:,0] == np.array([1, 1, 0, -1])).all()
    assert K.trace()==4
    assert np.isnan(K[[1, 1, 2, 2, 3, 3], [2, 3, 1, 3, 1, 2]]).all()
    assert (A[0,:] == np.array([1, 1, .5, 0])).all()
    assert (A[:,0] == np.array([1, 1, .5, 0])).all()
    assert A.trace()==4
    assert np.isnan(A[[1, 1, 2, 2, 3, 3], [2, 3, 1, 3, 1, 2]]).all()

#################################################################
#  UNIT TESTS
#################################################################
test_1()
test_2()
test_3()

# End
#################################################################