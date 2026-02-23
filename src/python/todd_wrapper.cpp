#include "algorithms.hpp"
#include "matrix.hpp"
#include "nullspace.hpp"
#include "random.hpp"
#include "todd_generator.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace todd;

template <class T> static std::string to_string_stream(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

static Row bitvec_from_list(std::size_t n, const std::vector<bool>& bits) {
    Row        bv(n);
    const auto k = std::min(n, bits.size());
    for (std::size_t i = 0; i < k; ++i)
        if (bits[i])
            bv.set(i);
    return bv;
}

static py::tuple candidate_to_tuple(const CandidateExport& c) {
    py::object tD = (c.tohpe_dim == -1) ? py::none() : py::object(py::int_(c.tohpe_dim));
    py::object k  = (c.k == k_single_sentinel<Int>()) ? py::none() : py::object(py::int_(c.k));
    py::object l  = (c.l == k_single_sentinel<Int>()) ? py::none() : py::object(py::int_(c.l));
    return py::make_tuple(c.pool_score, c.final_score, c.reduction, k, l, c.basis_dim, c.bucket_size, tD,
                          c.num_better_dim, c.num_better_red, c.num_better_pool_score);
}

struct PyTohpeGenerator {
    std::shared_ptr<MatrixWithData> data;
    TohpeGenerator                  gen;

    explicit PyTohpeGenerator(std::shared_ptr<MatrixWithData> d) : data(std::move(d)), gen(data) {}

    NullSpace     make(const Row& z) const { return gen.make(z); }
    NullSpace     make1(index_t row) const { return gen.make(row); }
    NullSpace     make2(index_t row1, index_t row2) const { return gen.make(row1, row2); }
    Row           best_z(const Row& y) const { return gen.best_z(y); }
    const Matrix& P() const noexcept { return data->P(); }
};

struct PyToddGenerator {
    std::shared_ptr<MatrixWithData> data;
    FullToddGenerator               gen;

    explicit PyToddGenerator(std::shared_ptr<MatrixWithData> d) : data(std::move(d)), gen(data) {}

    NullSpace     make(const Row& z) const { return gen.make(z); }
    NullSpace     make1(index_t row) const { return gen.make(row); }
    NullSpace     make2(index_t row1, index_t row2) const { return gen.make(row1, row2); }
    const Matrix& P() const noexcept { return data->P(); }
};

PYBIND11_MODULE(pyvartodd, m) {

    py::class_<Row>(m, "BitVec")
        .def(py::init<std::size_t>(), py::arg("n"))
        .def(py::init(&bitvec_from_list), py::arg("n"), py::arg("bits"))
        .def_static(
            "from_numpy",
            [](py::array_t<bool, py::array::forcecast> a) {
                if (a.ndim() != 1)
                    throw py::value_error("BitVec.from_numpy expects 1D array");

                Row  bv(static_cast<std::size_t>(a.shape(0)));
                auto r = a.unchecked<1>(); // safe element access
                for (py::ssize_t i = 0; i < r.shape(0); ++i)
                    if (r(i))
                        bv.set(static_cast<std::size_t>(i));
                return bv;
            },
            py::arg("a"))
        .def("to_numpy",
             [](const Row& b) {
                 py::array_t<bool> out(b.size());
                 auto              buf = out.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < b.size(); ++i)
                     buf(i) = b.test(i);
                 return out;
             })
        .def("__len__", &Row::size)
        .def("size", &Row::size)
        .def("count", &Row::count)
        .def("__getitem__",
             [](const Row& b, std::size_t i) {
                 if (i >= b.size())
                     throw py::index_error();
                 return b.test(i);
             })
        .def("__setitem__",
             [](Row& b, std::size_t i, bool v) {
                 if (i >= b.size())
                     throw py::index_error();
                 if (v)
                     b.set(i);
                 else
                     b.reset(i);
             })
        .def(
            "__xor__",
            [](const Row& a, const Row& b) {
                if (a.size() != b.size())
                    throw py::value_error("size mismatch");
                auto r = a;
                r ^= b;
                return r;
            },
            py::is_operator())
        .def(
            "__ixor__",
            [](Row& a, const Row& b) {
                if (a.size() != b.size())
                    throw py::value_error("size mismatch");
                a ^= b;
                return std::ref(a);
            },
            py::is_operator())
        .def("to_list",
             [](const Row& b) {
                 std::vector<bool> out(b.size());
                 for (std::size_t i = 0; i < b.size(); ++i)
                     out[i] = b.test(i);
                 return out;
             })
        .def("__repr__", [](const Row& b) { return to_string_stream(b); });

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<index_t, index_t>(), py::arg("rows"), py::arg("cols"))
        .def_static("zeros", &Matrix::zeros, py::arg("rows"), py::arg("cols"))
        .def_static("identity", &Matrix::identity, py::arg("n"))
        .def_static("from_rows", &Matrix::from_rows, py::arg("rows"))
        .def_static("from_numpy",
                    [](py::array_t<bool, py::array::c_style | py::array::forcecast> A) {
                        if (A.ndim() != 2)
                            throw py::value_error("Matrix.from_numpy expects 2D array");
                        Matrix M(A.shape(0), A.shape(1));
                        auto   buf = A.unchecked<2>();
                        for (index_t i = 0; i < M.rows(); ++i) {
                            for (index_t j = 0; j < M.cols(); ++j)
                                if (buf(i, j))
                                    M[i].set(j);
                        }
                        return M;
                    })
        .def("to_numpy",
             [](const Matrix& M) {
                 py::array_t<bool> out({M.rows(), M.cols()});
                 auto              buf = out.mutable_unchecked<2>();
                 for (index_t i = 0; i < M.rows(); ++i)
                     for (index_t j = 0; j < M.cols(); ++j)
                         buf(i, j) = M[i].test(j);
                 return out;
             })
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def("__len__", &Matrix::rows)
        .def("transpose", &Matrix::transpose)
        .def("append_right_inplace", &Matrix::append_right_inplace, py::return_value_policy::reference_internal)
        .def("append_down_inplace", &Matrix::append_down_inplace, py::return_value_policy::reference_internal)
        .def("__getitem__",
             [](const Matrix& M, index_t i) {
                 if (i >= M.rows())
                     throw py::index_error();
                 return Row(M[i]);
             })
        .def("__setitem__",
             [](Matrix& M, index_t i, const Row& r) {
                 if (i >= M.rows())
                     throw py::index_error();
                 if (M.cols() != r.size())
                     throw py::value_error("row size mismatch");
                 assign(M[i], r);
             })
        .def("__eq__", &Matrix::operator==, py::is_operator())
        .def("__ne__", &Matrix::operator!=, py::is_operator())
        .def(
            "__add__", [](const Matrix& A, const Matrix& B) { return A + B; }, py::is_operator())
        .def(
            "__iadd__",
            [](Matrix& A, const Matrix& B) {
                A += B;
                return std::ref(A);
            },
            py::is_operator())
        .def(
            "__mul__", [](const Matrix& A, const Matrix& B) { return A * B; }, py::is_operator())
        .def(
            "__matmul__", [](const Matrix& A, const Row& v) { return A * v; }, py::is_operator())
        .def(
            "seed",
            [](const Matrix& mat, std::uint32_t base_seed, std::uint32_t step) {
                return matrix_seed(mat, base_seed, step);
            },
            py::arg("base_seed") = 0U, py::arg("add_seed") = 0U)
        .def("__repr__", [](const Matrix& M) { return to_string_stream(M); })
        .def(py::pickle(
            // __getstate__
            [](const Matrix& M) {
                // Convert to numpy array and return it as the state
                py::array_t<bool> arr({M.rows(), M.cols()});
                auto              buf = arr.mutable_unchecked<2>();
                for (index_t i = 0; i < M.rows(); ++i) {
                    for (index_t j = 0; j < M.cols(); ++j) {
                        buf(i, j) = M[i].test(j);
                    }
                }
                return arr;
            },
            // __setstate__
            [](py::array_t<bool, 16> arr) {
                if (arr.ndim() != 2) {
                    throw std::runtime_error("Invalid state: expected 2D numpy array");
                }

                Matrix M(arr.shape(0), arr.shape(1));
                auto   buf = arr.unchecked<2>();
                for (index_t i = 0; i < M.rows(); ++i) {
                    for (index_t j = 0; j < M.cols(); ++j) {
                        if (buf(i, j)) {
                            M[i].set(j);
                        }
                    }
                }
                return M;
            }));
    py::class_<Tensor3D>(m, "Tensor3D")
        .def(py::init<index_t>())
        .def(py::init<const Matrix&>())
        .def_property_readonly("n", &Tensor3D::n)
        .def("get", &Tensor3D::get)
        .def("to_numpy",
             [](const Tensor3D& T) {
                 py::array_t<bool> out({T.n(), T.n(), T.n()});
                 auto              buf = out.mutable_unchecked<3>();
                 for (index_t k = 0; k < T.n(); ++k)
                     for (index_t j = 0; j < T.n(); ++j)
                         for (index_t i = 0; i < T.n(); ++i)
                             buf(i, j, k) = T.get(i, j, k);
                 return out;
             })
        .def("__eq__", &Tensor3D::operator==, py::is_operator())
        .def("__ne__", &Tensor3D::operator!=, py::is_operator());

    py::class_<MatrixWithData, std::shared_ptr<MatrixWithData>>(m, "MatrixWithData")
        .def(py::init([](Matrix P) { return std::make_shared<MatrixWithData>(std::move(P)); }), py::arg("P"))
        .def_property_readonly("P", &MatrixWithData::P, py::return_value_policy::reference_internal)
        .def_property_readonly("tohpe_basis", &MatrixWithData::tohpe_basis, py::return_value_policy::reference_internal)
        .def("tohpe_dim", [](const MatrixWithData& M) { return M.tohpe_basis().rows(); });

    py::class_<NullSpace>(m, "NullSpace")
        .def("apply", [](const NullSpace& self, const Row& row) { return self.apply(row); })
        .def("rank_divergence", [](const NullSpace& self, const Row& row) { return self.rank_divergence(row); })
        .def("linear_combination", [](const NullSpace& self, const Row& row) { return self.linear_combination(row); })
        .def_property_readonly("basis", &NullSpace::basis, py::return_value_policy::reference_internal)
        .def_property_readonly("P", &NullSpace::P, py::return_value_policy::reference_internal)
        .def_property_readonly("z", &NullSpace::vector, py::return_value_policy::reference_internal);

    py::class_<PyTohpeGenerator>(m, "TohpeGenerator")
        .def(py::init<std::shared_ptr<MatrixWithData>>(), py::arg("data"))
        .def("make", &PyTohpeGenerator::make, py::arg("z"), py::return_value_policy::move, py::keep_alive<0, 1>())
        .def("make1", &PyTohpeGenerator::make1, py::arg("row"), py::return_value_policy::move, py::keep_alive<0, 1>())
        .def("make2", &PyTohpeGenerator::make2, py::arg("row1"), py::arg("row2"), py::return_value_policy::move,
             py::keep_alive<0, 1>())
        .def("best_z", &PyTohpeGenerator::best_z)
        .def_property_readonly("P", &PyTohpeGenerator::P, py::return_value_policy::reference_internal);

    py::class_<PyToddGenerator>(m, "ToddGenerator")
        .def(py::init<std::shared_ptr<MatrixWithData>>(), py::arg("data"))
        .def("make", &PyToddGenerator::make, py::arg("z"), py::return_value_policy::move, py::keep_alive<0, 1>())
        .def("make1", &PyToddGenerator::make1, py::arg("row"), py::return_value_policy::move, py::keep_alive<0, 1>())
        .def("make2", &PyToddGenerator::make2, py::arg("row1"), py::arg("row2"), py::return_value_policy::move,
             py::keep_alive<0, 1>())
        .def_property_readonly("P", &PyToddGenerator::P, py::return_value_policy::reference_internal);

    py::class_<PyRNG>(m, "RNG")
        .def(py::init<>())
        .def(py::init<std::uint32_t>(), py::arg("seed"))
        .def("rand_int", &PyRNG::rand_int)
        .def("rand_double", &PyRNG::rand_double, py::arg("low") = 0.0, py::arg("high") = 1.0)
        .def("random", &PyRNG::random_raw)
        .def("seed", &PyRNG::seed)
        .def("sample_unique_bitvecs", &PyRNG::sample_small_unique_bitvectors)
        .def("sample_bitvec", &PyRNG::sample_bitvector);

    m.def(
        "gauss_elimination_inplace",
        [](Matrix& A, Matrix& aug, pivot_map& pivots) { gauss_elimination_inplace(A, aug, pivots); }, py::arg("A"),
        py::arg("aug"), py::arg("pivots") = pivot_map());
    m.def("extract_basis", &extract_basis);
    m.def("L_expansion", &L_expansion);

#ifdef _OPENMP
    m.def("set_omp_threads", [](int n) {
        if (n <= 0)
            throw std::runtime_error("threads must be > 0");
        omp_set_num_threads(n);
    });
    m.def("get_omp_threads", []() { return omp_get_max_threads(); });
#else
    m.def("set_omp_threads", [](int) {});
    m.def("get_omp_threads", []() { return 1; });
#endif

    py::class_<CandidateExport>(m, "CandidateExport")
        .def(py::init<>())
        .def_readwrite("pool_score", &CandidateExport::pool_score)
        .def_readwrite("final_score", &CandidateExport::final_score)
        .def_readwrite("reduction", &CandidateExport::reduction)
        .def_readwrite("k", &CandidateExport::k)
        .def_readwrite("l", &CandidateExport::l)
        .def_readwrite("basis_dim", &CandidateExport::basis_dim)
        .def_readwrite("tohpe_dim", &CandidateExport::tohpe_dim)
        .def_readwrite("bucket_size", &CandidateExport::bucket_size)
        .def_readwrite("num_better_dim", &CandidateExport::num_better_dim)
        .def_readwrite("num_better_red", &CandidateExport::num_better_red)
        .def_readwrite("num_better_pool_score", &CandidateExport::num_better_pool_score)
        .def("__repr__", [](const CandidateExport& c) {
            return "CandidateExport(score=" + std::to_string(c.pool_score) +
                   ", reduction=" + std::to_string((long long)c.reduction) + ", k=" + std::to_string((long long)c.k) +
                   ", l=" + std::to_string((long long)c.l) + ", basis_dim=" + std::to_string((long long)c.basis_dim) +
                   ", tohpe_dim=" + std::to_string((long long)c.tohpe_dim) +
                   ", bucket_size=" + std::to_string((long long)c.bucket_size) + ")";
        });

    // ---------------- Stats ----------------
    py::class_<Stats>(m, "Stats")
        .def(py::init<>())
        .def_readwrite("total", &Stats::total)
        .def_readwrite("nonzero", &Stats::nonzero)
        .def_readwrite("mean_mr", &Stats::mean_mr)
        .def_readwrite("mean_basis", &Stats::mean_basis)
        .def_readwrite("mean_reduction", &Stats::mean_reduction)
        .def_readwrite("mean_score", &Stats::mean_score)
        .def_readwrite("accepted", &Stats::accepted)
        .def_readwrite("accepted_non_improving", &Stats::accepted_non_improving)
        .def_readwrite("accepted_tohpe", &Stats::accepted_tohpe)
        .def_readwrite("max_tohpe_dim", &Stats::max_final_tohpe_dim)
        .def_readwrite("mean_tohpe_dim", &Stats::mean_final_tohpe_dim)
        .def_readwrite("max_score", &Stats::max_pool_score)
        .def_readwrite("max_basis", &Stats::max_basis)
        .def_readwrite("max_reduction", &Stats::max_reduction)
        .def_readwrite("max_bucket", &Stats::max_bucket)
        .def("__repr__", [](const Stats& s) {
            return "Stats(total=" + std::to_string((unsigned long long)s.total) +
                   ", nonzero=" + std::to_string((unsigned long long)s.nonzero) +
                   ", mean_score=" + std::to_string(s.mean_score) + ", max_score=" + std::to_string(s.max_pool_score) +
                   ", max_basis=" + std::to_string((long long)s.max_basis) +
                   ", max_reduction=" + std::to_string((long long)s.max_reduction) + ")";
        });
    // ---------------- Result ----------------
    py::class_<ExplorationScore>(m, "ExplorationScore")
        // .def(py::init<float, float, float>(), py::arg("wred") = 1.0, py::arg("wdim") = 1.0, py::arg("wbucket") = 1.0)
        .def(py::init<float, float, float, float, float>(), py::arg("wred") = 1.0, py::arg("wdim") = 1.0,
             py::arg("wbucket") = 1.0, py::arg("wvw") = 1.0, py::arg("wz") = 1.0)
        .def("__repr__",
             [](const ExplorationScore& r) {
                 return "ExplorationScore(wdim=" + std::to_string(r.wdim) + "wbucket=" + std::to_string(r.wbucket) +
                        "wred=" + std::to_string(r.wred) + "wvw=" + std::to_string(r.wz) + "wvw=" + std::to_string(r.wz) + ")";
             })
        .def("__str__",
             [](const ExplorationScore& r) {
                 return "(" + std::to_string(r.wdim) + ", " + std::to_string(r.wbucket) + ", " +
                        std::to_string(r.wred) + ", " + std::to_string(r.wvw) + ", " + std::to_string(r.wz) + ")";
             })
        .def("__len__",
             [](const ExplorationScore& r) {
                 // Check if wvw has been set (you need a way to detect this)
                 // For simplicity, always return 4 or add a member variable to track
                 return 5;
             })
        .def("__getitem__",
             [](const ExplorationScore& r, py::ssize_t index) {
                 if (index < 0 || index >= 5) {
                     throw py::index_error("ExplorationScore index out of range");
                 }
                 switch (index) {
                 case 0:
                     return py::float_(r.wred);
                 case 1:
                     return py::float_(r.wdim);
                 case 2:
                     return py::float_(r.wbucket);
                 case 3: {
                     return py::float_(r.wvw);
                 }
                case 4: {
                     return py::float_(r.wz);
                 }
                 default:
                     return py::float_(0.0f);
                 }
             })
        .def(py::pickle([](const ExplorationScore& r) { return py::make_tuple(r.wred, r.wdim, r.wbucket, r.wvw, r.wz); },
                        [](py::tuple t) {
                            if (t.size() != 5 )
                                throw std::runtime_error("Invalid state for ExplorationScore!");

                            float wred    = t[0].cast<float>();
                            float wdim    = t[1].cast<float>();
                            float wbucket = t[2].cast<float>();
                            float wvw = t[3].cast<float>();
                            float wz = t[4].cast<float>();

                            return ExplorationScore(wred, wdim, wbucket, wvw, wz);
                        }));

    py::class_<FinalizationScore>(m, "FinalizationScore")
        // .def(py::init<float, float, float, float>(), py::arg("wred") = 1.0, py::arg("wdim") = 1.0,
        //      py::arg("wbucket") = 1.0, py::arg("wtohpe") = 1.0)
        .def(py::init<float, float, float, float, float, float>(), py::arg("wred") = 1.0, py::arg("wdim") = 1.0,
             py::arg("wbucket") = 1.0, py::arg("wvw") = 1.0, py::arg("wz") = 1.0, py::arg("wtohpe") = 1.0)
        .def("__repr__",
             [](const FinalizationScore& r) {
                 return "FinalizationScore(wdim=" + std::to_string(r.wdim) + "wbucket=" + std::to_string(r.wbucket) +
                        "wred=" + std::to_string(r.wred) +  "wvw=" + std::to_string(r.wvw) +  "wz=" + std::to_string(r.wz) + "wtohpe=" + std::to_string(r.wtohpe_dim) + ")" 
                        ;
             })
        .def("__str__",
             [](const FinalizationScore& r) {
                 return "(" + std::to_string(r.wdim) + ", " + std::to_string(r.wbucket) + ", " +
                        std::to_string(r.wred) + ", " + std::to_string(r.wtohpe_dim) + ", " + std::to_string(r.wvw) + std::to_string(r.wz) +
                        ")";
             })
        .def("__len__", [](const FinalizationScore& r) { return 5; })
        .def("__getitem__",
             [](const FinalizationScore& r, py::ssize_t index) {
                 if (index < 0 || index >= 6) {
                     throw py::index_error("FinalizationScore index out of range");
                 }
                 switch (index) {
                 case 0:
                     return py::float_(r.wred);
                 case 1:
                     return py::float_(r.wdim);
                 case 2:
                     return py::float_(r.wbucket);
                 case 3:
                     return py::float_(r.wvw);
                 case 4:
                     return py::float_(r.wz);
                 case 5: {
                     return py::float_(r.wtohpe_dim);
                 }
                 default:
                     return py::float_(0.0f);
                 }
             })
        .def(py::pickle(
            [](const FinalizationScore& r) { return py::make_tuple(r.wred, r.wdim, r.wbucket, r.wvw, r.wz, r.wtohpe_dim ); },
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state for FinalizationScore!");

                float wred    = t[0].cast<float>();
                float wdim    = t[1].cast<float>();
                float wbucket = t[2].cast<float>();
                float wvw     = t[3].cast<float>();
                float wz     = t[4].cast<float>();
                float wtohpe  = t[5].cast<float>();

                return FinalizationScore(wred, wdim, wbucket, wvw, wz, wtohpe);
            }));

    // ---------------- Result ----------------
    py::class_<Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("chosen", &Result::chosen)
        .def_readwrite("states", &Result::states)
        .def_readwrite("stats", &Result::stats) // remove if Stats not bound
        .def_readwrite("seed", &Result::seed)
        .def("__repr__", [](const Result& r) {
            return "Result(chosen=" + std::to_string(r.chosen.size()) + ", states=" + std::to_string(r.states.size()) +
                   ", seed=" + std::to_string((unsigned long long)r.seed) + ")";
        });

    // ---------------- PolicyConfig ----------------
    // Expose as a Python-friendly config object (like a dataclass).
    py::class_<PolicyConfig>(m, "PolicyConfig")
        .def(py::init<>())
        .def(py::init<ExplorationScore, FinalizationScore, std::string, float, float, float, Int, Int, Int, Int, Int,
                      Int, Int, Int, Int, Int, Int, bool>(),
             py::arg("ExplorationScore") = ExplorationScore(), 
             py::arg("FinalizationScore") = FinalizationScore(),
             py::arg("selection") = "greedy",
             py::arg("temperature") = 0.0f, 
             py::arg("non_improving_prob") = 0.0f,
             py::arg("max_z_to_research_fraction") = 1.0f, 
             py::arg("num_samples") = 64, 
             py::arg("num_candidates") = 1,
             py::arg("top_pool") = 1, 
             py::arg("max_from_single_ns") = 10, 
             py::arg("min_reduction") = 0,
             py::arg("max_reduction") = k_single_sentinel<Int>(), 
             py::arg("max_z_to_research") = 1 << 30,
             py::arg("min_pool_size") = 1, 
             py::arg("max_tohpe") = 1, 
             py::arg("threads") = 1,
             py::arg("tohpe_sample") = 1, 
             py::arg("try_only_tohpe") = true)
        .def_readwrite("num_samples", &PolicyConfig::num_samples)
        .def_readwrite("num_candidates", &PolicyConfig::num_candidates)
        .def_readwrite("top_pool", &PolicyConfig::top_pool)
        .def_readwrite("selection", &PolicyConfig::selection)
        .def_readwrite("temperature", &PolicyConfig::temperature)
        .def_readwrite("non_improving_prob", &PolicyConfig::non_improving_prob)
        .def_readwrite("max_from_single_ns", &PolicyConfig::max_from_single_ns)
        .def_readwrite("min_reduction", &PolicyConfig::min_reduction)
        .def_readwrite("max_reduction", &PolicyConfig::max_reduction)
        .def_readwrite("min_pool_size", &PolicyConfig::min_pool_size)
        .def_readwrite("max_z_to_research", &PolicyConfig::max_z_to_research)
        .def_readwrite("max_z_to_research_fraction", &PolicyConfig::max_z_to_research_fraction)
        .def_readwrite("max_tohpe", &PolicyConfig::max_tohpe)
        .def_readwrite("threads", &PolicyConfig::threads)
        .def_readwrite("try_only_tohpe", &PolicyConfig::try_only_tohpe)
        .def_readwrite("tohpe_sample", &PolicyConfig::tohpe_sample)
        .def("__repr__", [](const PolicyConfig& c) {
            return "PolicyConfig(num_samples=" + std::to_string((long long)c.num_samples) +
                   ", num_candidates=" + std::to_string((long long)c.num_candidates) +
                   ", top_pool=" + std::to_string((long long)c.top_pool) + ", selection='" + c.selection + "'" +
                   ", temperature=" + std::to_string(c.temperature) +
                   ", non_improving_prob=" + std::to_string((long long)c.non_improving_prob) +
                   ", max_from_single_ns=" + std::to_string((long long)c.max_from_single_ns) +
                   ", min_reduction=" + std::to_string((long long)c.min_reduction) +
                   ", max_reduction=" + std::to_string((long long)c.max_reduction) +
                   ", min_pool_size=" + std::to_string((long long)c.min_pool_size) +
                   ", max_z_to_research=" + std::to_string((long long)c.max_z_to_research) +
                   ", max_z_to_research_fraction=" + std::to_string((long long)c.max_z_to_research_fraction) +
                   ", max_tohpe=" + std::to_string((long long)c.max_tohpe) +
                   ", threads=" + std::to_string((long long)c.threads) +
                   ", try_only_tohpe=" + std::to_string((long long)c.try_only_tohpe) +
                   ", tohpe_sample=" + std::to_string((long long)c.tohpe_sample) + ")";
        });

    m.def(
        "policy_iteration",
        [](Matrix cur_mat, PolicyConfig pcfg, index_t seed, index_t add_seed) {
            auto data = std::make_shared<MatrixWithData>(std::move(cur_mat));
            return policy_iteration_impl(data, pcfg, seed, add_seed);
        },
        py::arg("cur_mat"), py::arg("policy_cfg") = PolicyConfig{}, py::arg("seed") = 21, py::arg("add_seed") = 0);
}
