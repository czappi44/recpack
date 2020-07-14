import numpy
import random
import scipy.sparse
import recpack.algorithms


def generate_data():
    # TODO generate scipy.sparse matrix with user interactions.
    users = list(range(10))
    u_, i_ = [], []
    for user in users:
        items = list(range(10))
        items_interacted = numpy.random.choice(items, 3, replace=False)

        u_.extend([user] * 3)
        i_.extend(items_interacted)
    return scipy.sparse.csr_matrix((numpy.ones(len(u_)), (u_, i_)))


def generate_in_out():
    for i in range(1, 11):
        users = list(range(i))
        u_in, i_in = [], []
        u_out, i_out = [], []
        for user in users:
            items = list(range(10))
            items_interacted = numpy.random.choice(items, 6, replace=False)
            items_in = items_interacted[:3]
            items_out = items_interacted[3:]

            u_in.extend([user] * 3)
            u_out.extend(([user] * 3))
            i_in.extend(items_in)
            i_out.extend(items_out)

        in_ = scipy.sparse.csr_matrix(
            (numpy.ones(len(u_in)), (u_in, i_in)), shape=(i, 10)
        )
        out_ = scipy.sparse.csr_matrix(
            (numpy.ones(len(u_out)), (u_out, i_out)), shape=(i, 10)
        )

        yield in_, out_


def test_random():
    train_data = generate_data()

    seed = 42
    K = 5
    algo = recpack.algorithms.algorithm_registry.get("random")(K=K, seed=42)
    algo.fit(train_data)

    for out_, in_ in generate_in_out():
        result = algo.predict(in_)
        assert len(result.nonzero()[1]) == result.shape[0] * K
        # TODO: What else to test?


def test_popularity():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = scipy.sparse.csr_matrix((values, (user_i, item_i)))
    algo = recpack.algorithms.algorithm_registry.get("popularity")(K=2)

    algo.fit(train_data)

    _in = scipy.sparse.csr_matrix(([1, 1], ([0, 1], [1, 1])), shape=(5, 5))
    prediction = algo.predict(_in)

    assert (prediction[0] != prediction[1]).nnz == 0
    assert prediction[0, 4] != 0
    assert prediction[0, 3] != 0
    assert prediction[0, 4] > prediction[0, 3]
    assert (prediction[0, :3].toarray() == numpy.array([0, 0, 0])).all()


def test_popularity_add():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = scipy.sparse.csr_matrix((values, (user_i, item_i)))
    algo = recpack.algorithms.algorithm_registry.get("popularity")(K=2)
    algo_2 = recpack.algorithms.algorithm_registry.get("popularity")(K=2)

    algo.fit(train_data)
    print(algo.sorted_scores_)
    algo_2.fit(train_data)
    print(algo_2.sorted_scores_)
    
    algo.add(algo_2)

    assert algo.sorted_scores_[0] == (algo_2.sorted_scores_[0][0], 2*algo_2.sorted_scores_[0][1])
