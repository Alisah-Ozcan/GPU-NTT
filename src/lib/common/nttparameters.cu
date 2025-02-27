// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "nttparameters.cuh"

namespace gpuntt
{

    int bitreverse(int index, int n_power)
    {
        int res_1 = 0;
        for (int i = 0; i < n_power; i++)
        {
            res_1 <<= 1;
            res_1 = (index & 1) | res_1;
            index >>= 1;
        }
        return res_1;
    }

    template <typename T>
    NTTParameters<T>::NTTParameters(int LOGN,
                                    ReductionPolynomial poly_reduce_type)
    {
        logn = LOGN;
        n = 1 << logn;

        poly_reduction = poly_reduce_type;

        modulus = modulus_pool();

        omega = omega_pool();
        psi = psi_pool();

        root_of_unity =
            (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
        inverse_root_of_unity = OPERATOR<T>::modinv(root_of_unity, modulus);

        root_of_unity_size =
            (poly_reduce_type == ReductionPolynomial::X_N_minus)
                ? (1 << (logn - 1))
                : (1 << logn);

        forward_root_of_unity_table_generator();
        inverse_root_of_unity_table_generator();

        n_inverse_generator();
    }

    template <typename T>
    NTTParameters<T>::NTTParameters(int LOGN, NTTFactors<T> ntt_factors,
                                    ReductionPolynomial poly_reduce_type)
    {
        logn = LOGN;
        n = 1 << logn;

        poly_reduction = poly_reduce_type;

        modulus = ntt_factors.modulus;

        omega = ntt_factors.omega;
        psi = ntt_factors.psi;

        n_inverse_generator();

        root_of_unity =
            (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
        inverse_root_of_unity = OPERATOR<T>::modinv(root_of_unity, modulus);

        root_of_unity_size =
            (poly_reduce_type == ReductionPolynomial::X_N_minus)
                ? (1 << (logn - 1))
                : (1 << logn);

        forward_root_of_unity_table_generator();
        inverse_root_of_unity_table_generator();

        n_inverse_generator();
    }

    template <typename T> NTTParameters<T>::NTTParameters(){};

    template <typename T> Modulus<T> NTTParameters<T>::modulus_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 primes[] = {
                268460033, 268582913, 268664833, 268369921, 269221889,
                269221889, 270532609, 270532609, 270532609, 377487361,
                377487361, 469762049, 469762049};

            Modulus32 prime(primes[logn - 12]);
            return prime;
        }
        else
        {
            static Data64 primes[] = {
                576460752303415297, 576460752303439873, 576460752304439297,
                576460752308273153, 576460752308273153, 576460752315482113,
                576460752315482113, 576460752340123649, 576460752364240897,
                576460752475389953, 576460752597024769, 576460753024843777,
                576460753175838721, 288230377292562433, 288230383802122241,
                288230385815388161, 288230385815388161};

            Modulus64 prime(primes[logn - 12]);
            return prime;
        }
    }

    template <typename T> T NTTParameters<T>::omega_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 W[] = {36747374, 249229369, 4092529, 175218169,
                                 10653696, 238764304, 240100,  23104,
                                 179776,   19321,     38809,   1600,
                                 169};

            return W[logn - 12];
        }
        else
        {
            static Data64 W[] = {
                288482366111684746, 37048445140799662,  459782973201979845,
                64800917766465203,  425015386842055933, 18734847765732801,
                119109113519742895, 227584740857897520, 477282059544659462,
                570131728462077067, 433594414095420776, 219263994987749328,
                189790554094222112, 96649110792683523,  250648942594717784,
                279172744045218282, 225865349704673648};

            return W[logn - 12];
        }
    }

    template <typename T> T NTTParameters<T>::psi_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 PSI[] = {77090, 15787, 2023, 13237, 3264, 15452, 490,
                                   152,   424,   139,  197,   40,   13};

            return PSI[logn - 12];
        }
        else
        {
            static Data64 PSI[] = {
                238394956950829, 54612008597396, 8242615629351, 16141297350887,
                3760097055997,   11571974431275, 328867687796,  2298846063117,
                731868219707,    409596963254,   189266227206,  31864818375,
                92067739764,     5214432335,     734084005,     3351406780,
                717004697};

            return PSI[logn - 12];
        }
    }

    template <typename T>
    void NTTParameters<T>::forward_root_of_unity_table_generator()
    {
        forward_root_of_unity_table.push_back(1);

        for (int i = 1; i < root_of_unity_size; i++)
        {
            T exp = OPERATOR<T>::mult(forward_root_of_unity_table[i - 1],
                                      root_of_unity, modulus);
            forward_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T>
    void NTTParameters<T>::inverse_root_of_unity_table_generator()
    {
        inverse_root_of_unity_table.push_back(1);

        for (int i = 1; i < root_of_unity_size; i++)
        {
            T exp = OPERATOR<T>::mult(inverse_root_of_unity_table[i - 1],
                                      inverse_root_of_unity, modulus);
            inverse_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T> void NTTParameters<T>::n_inverse_generator()
    {
        n_inv = OPERATOR<T>::modinv(n, modulus);
    }

    template <typename T>
    std::vector<Root<T>>
    NTTParameters<T>::gpu_root_of_unity_table_generator(std::vector<T> table)
    {
        // Taking Bitreverse order of root of unity table

        std::vector<T> new_table;
        int lg = log2(root_of_unity_size);
        for (int i = 0; i < root_of_unity_size; i++)
        {
            new_table.push_back(table[bitreverse(i, lg)]);
        }

        return new_table;
    }

    template <typename T>
    NTTParameters4Step<T>::NTTParameters4Step(
        int LOGN, ReductionPolynomial poly_reduce_type)
    {
        logn = LOGN;
        n = 1 << logn;

        poly_reduction = poly_reduce_type;

        modulus = modulus_pool();

        omega = omega_pool();
        psi = psi_pool();

        root_of_unity =
            (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
        inverse_root_of_unity = OPERATOR<T>::modinv(root_of_unity, modulus);

        root_of_unity_size =
            (poly_reduce_type == ReductionPolynomial::X_N_minus)
                ? (1 << (logn - 1))
                : (1 << logn);

        std::vector<int> dimention = matrix_dimention();
        n1 = dimention[0];
        n2 = dimention[1];

        small_forward_root_of_unity_table_generator();
        small_inverse_root_of_unity_table_generator();

        TW_forward_table_generator();
        TW_inverse_table_generator();

        n_inverse_generator();
    }

    template <typename T> NTTParameters4Step<T>::NTTParameters4Step(){};

    template <typename T> Modulus<T> NTTParameters4Step<T>::modulus_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 primes[] = {
                268460033, 268582913, 268664833, 268369921, 269221889,
                269221889, 270532609, 270532609, 270532609, 377487361,
                377487361, 469762049, 469762049};

            Modulus32 prime(primes[logn - 12]);
            return prime;
        }
        else
        {
            static Data64 primes[] = {
                576460752303415297, 576460752303439873, 576460752304439297,
                576460752308273153, 576460752308273153, 576460752315482113,
                576460752315482113, 576460752340123649, 576460752364240897,
                576460752475389953, 576460752597024769, 576460753024843777,
                576460753175838721, 288230377292562433, 288230383802122241,
                288230385815388161, 288230385815388161};

            Modulus64 prime(primes[logn - 12]);
            return prime;
        }
    }
    template <typename T> T NTTParameters4Step<T>::omega_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 W[] = {36747374, 249229369, 4092529, 175218169,
                                 10653696, 238764304, 240100,  23104,
                                 179776,   19321,     38809,   1600,
                                 169};

            return W[logn - 12];
        }
        else
        {
            static Data64 W[] = {
                288482366111684746, 37048445140799662,  459782973201979845,
                64800917766465203,  425015386842055933, 18734847765732801,
                119109113519742895, 227584740857897520, 477282059544659462,
                570131728462077067, 433594414095420776, 219263994987749328,
                189790554094222112, 96649110792683523,  250648942594717784,
                279172744045218282, 225865349704673648};

            return W[logn - 12];
        }
    }

    template <typename T> T NTTParameters4Step<T>::psi_pool()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        if constexpr (std::is_same<T, Data32>::value)
        {
            static Data32 PSI[] = {77090, 15787, 2023, 13237, 3264, 15452, 490,
                                   152,   424,   139,  197,   40,   13};

            return PSI[logn - 12];
        }
        else
        {
            static Data64 PSI[] = {
                238394956950829, 54612008597396, 8242615629351, 16141297350887,
                3760097055997,   11571974431275, 328867687796,  2298846063117,
                731868219707,    409596963254,   189266227206,  31864818375,
                92067739764,     5214432335,     734084005,     3351406780,
                717004697};

            return PSI[logn - 12];
        }
    }

    template <typename T>
    std::vector<int> NTTParameters4Step<T>::matrix_dimention()
    {
        customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
        std::vector<int> shape;
        switch (logn)
        {
            case 12:
                shape = {32, 128};
                return shape;
            case 13:
                shape = {32, 256};
                return shape;
            case 14:
                shape = {32, 512};
                return shape;
            case 15:
                shape = {64, 512};
                return shape;
            case 16:
                shape = {128, 512};
                return shape;
            case 17:
                shape = {32, 4096};
                return shape;
            case 18:
                shape = {32, 8192};
                return shape;
            case 19:
                shape = {32, 16384};
                return shape;
            case 20:
                shape = {32, 32768};
                return shape;
            case 21:
                shape = {64, 32768};
                return shape;
            case 22:
                shape = {128, 32768};
                return shape;
            case 23:
                shape = {128, 65536};
                return shape;
            case 24:
                shape = {256, 65536};
                return shape;
            default:
                throw std::runtime_error("Invalid choice.\n");
        }
    }

    template <typename T>
    void NTTParameters4Step<T>::small_forward_root_of_unity_table_generator()
    {
        T exp_n1 = int(n / n1);
        T small_root_of_unity_n1 =
            OPERATOR<T>::exp(root_of_unity, exp_n1, modulus);
        n1_based_root_of_unity_table.push_back(1);
        for (int i = 1; i < (n1 >> 1); i++)
        {
            T exp = OPERATOR<T>::mult(n1_based_root_of_unity_table[i - 1],
                                      small_root_of_unity_n1, modulus);
            n1_based_root_of_unity_table.push_back(exp);
        }

        T exp_n2 = int(n / n2);
        T small_root_of_unity_n2 =
            OPERATOR<T>::exp(root_of_unity, exp_n2, modulus);
        n2_based_root_of_unity_table.push_back(1);
        for (int i = 1; i < (n2 >> 1); i++)
        {
            T exp = OPERATOR<T>::mult(n2_based_root_of_unity_table[i - 1],
                                      small_root_of_unity_n2, modulus);
            n2_based_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T>
    void NTTParameters4Step<T>::TW_forward_table_generator()
    {
        int lg = log2(n1);
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                T index = bitreverse(i, lg);
                index = index * j;
                W_root_of_unity_table.push_back(
                    OPERATOR<T>::exp(root_of_unity, index, modulus));
            }
        }
    }

    template <typename T>
    void NTTParameters4Step<T>::small_inverse_root_of_unity_table_generator()
    {
        T exp_n1 = int(n / n1);
        T small_root_of_unity_n1 =
            OPERATOR<T>::exp(root_of_unity, exp_n1, modulus);
        small_root_of_unity_n1 =
            OPERATOR<T>::modinv(small_root_of_unity_n1, modulus);
        n1_based_inverse_root_of_unity_table.push_back(1);
        for (int i = 1; i < (n1 >> 1); i++)
        {
            T exp =
                OPERATOR<T>::mult(n1_based_inverse_root_of_unity_table[i - 1],
                                  small_root_of_unity_n1, modulus);
            n1_based_inverse_root_of_unity_table.push_back(exp);
        }

        T exp_n2 = int(n / n2);
        T small_root_of_unity_n2 =
            OPERATOR<T>::exp(root_of_unity, exp_n2, modulus);
        small_root_of_unity_n2 =
            OPERATOR<T>::modinv(small_root_of_unity_n2, modulus);
        n2_based_inverse_root_of_unity_table.push_back(1);
        for (int i = 1; i < (n2 >> 1); i++)
        {
            T exp =
                OPERATOR<T>::mult(n2_based_inverse_root_of_unity_table[i - 1],
                                  small_root_of_unity_n2, modulus);
            n2_based_inverse_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T>
    void NTTParameters4Step<T>::TW_inverse_table_generator()
    {
        int lg = log2(n2);
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                T index = bitreverse(j, lg);
                index = index * i;
                W_inverse_root_of_unity_table.push_back(
                    OPERATOR<T>::exp(inverse_root_of_unity, index, modulus));
            }
        }
    }

    template <typename T> void NTTParameters4Step<T>::n_inverse_generator()
    {
        n_inv = OPERATOR<T>::modinv(n, modulus);
    }

    template <typename T> void NTTParameters4Step<T>::n_inverse_generator_gpu()
    {
        n_inv_gpu = OPERATOR<T>::modinv(n, modulus);
    }

    template <typename T>
    std::vector<Root<T>>
    NTTParameters4Step<T>::gpu_root_of_unity_table_generator(
        std::vector<T> table)
    {
        // Taking Bitreverse order of root of unity table

        std::vector<T> new_table;
        int lg = log2(table.size());
        for (int i = 0; i < table.size(); i++)
        {
            new_table.push_back(table[bitreverse(i, lg)]);
        }

        return new_table;
    }

    template class NTTParameters<Data32>;
    template class NTTParameters<Data64>;
    template class NTTParameters4Step<Data32>;
    template class NTTParameters4Step<Data64>;

} // namespace gpuntt