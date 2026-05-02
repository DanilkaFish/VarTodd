#include "matrix.hpp"
#include "nullspace.hpp"
#include "random.hpp"
#include "todd_generator.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


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
        [](const Matrix& M) {
            py::array_t<bool> arr({M.rows(), M.cols()});
            auto              buf = arr.mutable_unchecked<2>();
            for (index_t i = 0; i < M.rows(); ++i) {
                for (index_t j = 0; j < M.cols(); ++j) {
                    buf(i, j) = M[i].test(j);
                }
            }
            return arr;
        },
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
    .def("sample_unique_bitvecs", &PyRNG::sample_unique_bitvectors)
    .def("sample_bitvec", &PyRNG::sample_bitvector);

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
    })
    .def(py::pickle(
        [](const CandidateExport& c) {
            return py::make_tuple(
                c.pool_score,
                c.final_score,
                c.reduction,
                c.k,
                c.l,
                c.basis_dim,
                c.tohpe_dim,
                c.bucket_size,
                c.num_better_dim,
                c.num_better_red,
                c.num_better_pool_score
            );
        },
        [](py::tuple t) { 
            if (t.size() != 11)
                throw std::runtime_error("Invalid state!");

            CandidateExport c;
            
            c.pool_score = t[0].cast<decltype(c.pool_score)>();
            c.final_score = t[1].cast<decltype(c.final_score)>();
            c.reduction = t[2].cast<decltype(c.reduction)>();
            c.k = t[3].cast<decltype(c.k)>();
            c.l = t[4].cast<decltype(c.l)>();
            c.basis_dim = t[5].cast<decltype(c.basis_dim)>();
            c.tohpe_dim = t[6].cast<decltype(c.tohpe_dim)>();
            c.bucket_size = t[7].cast<decltype(c.bucket_size)>();
            c.num_better_dim = t[8].cast<decltype(c.num_better_dim)>();
            c.num_better_red = t[9].cast<decltype(c.num_better_red)>();
            c.num_better_pool_score = t[10].cast<decltype(c.num_better_pool_score)>();
            
            return c;
        }
    ));

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
    })
    .def(py::pickle(
        [](const Stats& s) {
            return py::make_tuple(
                s.total,
                s.nonzero,
                s.mean_mr,
                s.mean_basis,
                s.mean_reduction,
                s.mean_score,
                s.accepted,
                s.accepted_non_improving,
                s.accepted_tohpe,
                s.max_final_tohpe_dim,
                s.mean_final_tohpe_dim,
                s.max_pool_score,
                s.max_basis,
                s.max_reduction,
                s.max_bucket
            );
        },
        [](py::tuple t) {
            if (t.size() != 15)
                throw std::runtime_error("Invalid state!");

            Stats s;
            
            s.total = t[0].cast<decltype(s.total)>();
            s.nonzero = t[1].cast<decltype(s.nonzero)>();
            s.mean_mr = t[2].cast<decltype(s.mean_mr)>();
            s.mean_basis = t[3].cast<decltype(s.mean_basis)>();
            s.mean_reduction = t[4].cast<decltype(s.mean_reduction)>();
            s.mean_score = t[5].cast<decltype(s.mean_score)>();
            s.accepted = t[6].cast<decltype(s.accepted)>();
            s.accepted_non_improving = t[7].cast<decltype(s.accepted_non_improving)>();
            s.accepted_tohpe = t[8].cast<decltype(s.accepted_tohpe)>();
            s.max_final_tohpe_dim = t[9].cast<decltype(s.max_final_tohpe_dim)>();
            s.mean_final_tohpe_dim = t[10].cast<decltype(s.mean_final_tohpe_dim)>();
            s.max_pool_score = t[11].cast<decltype(s.max_pool_score)>();
            s.max_basis = t[12].cast<decltype(s.max_basis)>();
            s.max_reduction = t[13].cast<decltype(s.max_reduction)>();
            s.max_bucket = t[14].cast<decltype(s.max_bucket)>();
            
            return s;
        }
    ));
    // ---------------- Result ----------------
    py::enum_<ScoringFunction::Function>(m, "Function")
        .value("LINEAR", ScoringFunction::LINEAR)
        .value("POLYNOM", ScoringFunction::POLYNOM)
        .value("DISTANCE", ScoringFunction::DISTANCE)
        .value("LOGARITHMIC", ScoringFunction::LOGARITHMIC)
        .value("SIGMOID", ScoringFunction::SIGMOID)
        .export_values();

    // ScoringFunction class
    py::class_<ScoringFunction>(m, "ScoringFunction")
        .def(py::init<>())
        .def(py::init([](ScoringFunction::Function type, float pow) {
            ScoringFunction sf;
            sf.type = type;
            sf.pow = pow;
            return sf;
        }), py::arg("type") = ScoringFunction::LINEAR, py::arg("pow") = 2.0f)
        .def_readwrite("type", &ScoringFunction::type)
        .def_readwrite("pow", &ScoringFunction::pow)

        .def(py::pickle(
            [](const ScoringFunction& sf) { return py::make_tuple(static_cast<int>(sf.type), sf.pow);; },
            [](py::tuple t) { 
                if (t.size() != 2) {
                    throw std::runtime_error("Invalid state!");
                }
                ScoringFunction sf;
                sf.type = static_cast<ScoringFunction::Function>(t[0].cast<int>());
                sf.pow = t[1].cast<float>();
                return sf; }
            ));

py::class_<ExplorationScore>(m, "ExplorationScore")
    .def(py::init<const std::vector<float>&, const std::vector<float>&, const ScoringFunction&>(),
             py::arg("weights") = std::vector<float>{0,0,0,0,0},
             py::arg("centers") = std::vector<float>{0,0,0,0,0},
             py::arg("sc") = ScoringFunction())
    .def(py::init<const std::vector<float>&, const std::vector<float>&,float>(),
        py::arg("weights") = std::vector<float>{0,0,0,0,0},
        py::arg("centers") = std::vector<float>{0,0,0,0,0},
        py::arg("pow")=1.0f)
    .def("__len__", [](const ExplorationScore&) { return 5; })
    .def("pow", [](const ExplorationScore& es) { return es.sc.pow; })
    
    .def("__getitem__",
         [](const ExplorationScore& r, py::ssize_t index) {
             if (index < 0 || index >= 10) {
                 throw py::index_error("ExplorationScore weights index out of range");
             }
             if (index <= 4)
                return py::float_(r.weights[index]);
             return py::float_(r.centers[index - 5]);
         })
    
    .def(py::pickle(
        [](const ExplorationScore& r) { 
            return py::make_tuple(r.weights, r.centers, r.sc);
        },
        [](py::tuple t) {
            if (t.size() != 3)
                throw std::runtime_error("Invalid state for ExplorationScore! Expected 6 elements.");
            ExplorationScore es(
                t[0].cast<std::vector<float>>(),
                t[1].cast<std::vector<float>>(),
                t[2].cast<ScoringFunction>()
            );
            return es;
        }
    ));

py::class_<FinalizationScore>(m, "FinalizationScore")
    .def(py::init<const std::vector<float>&, const std::vector<float>&, const ScoringFunction&>(),
             py::arg("weights") = std::vector<float>{0,0,0,0,0,0},
             py::arg("centers") = std::vector<float>{0,0,0,0,0,0},
             py::arg("sc") = ScoringFunction())
    .def(py::init<const std::vector<float>&, const std::vector<float>&,float>(),
            py::arg("weights") = std::vector<float>{0,0,0,0,0,0},
            py::arg("centers") = std::vector<float>{0,0,0,0,0,0},
            py::arg("pow")=1.0f)
    .def("pow", [](const FinalizationScore& fs) { return fs.sc.pow; })
    .def("__len__", [](const FinalizationScore&) { return 6; })
    
    .def("__getitem__",
         [](const FinalizationScore& r, py::ssize_t index) {
             if (index < 0 || index >= 12) {
                 throw py::index_error("FinalizationScore index out of range");
             }
             if (index <= 5)
                return py::float_(r.weights[index]);
             return py::float_(r.centers[index - 6]);
         })
    
    .def(py::pickle(
        [](const FinalizationScore& r) { 
            return py::make_tuple(r.weights, r.centers, r.sc);
        },
        [](py::tuple t) {
            if (t.size() != 3) {
                throw std::runtime_error("Invalid state for FinalizationScore Expected 3 elements.");
            }
            FinalizationScore fs(
                t[0].cast<std::vector<float>>(),
                t[1].cast<std::vector<float>>(),
                t[2].cast<ScoringFunction>()
            );
            return fs;
        }
    ));
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

    py::class_<PolicyConfig>(m, "PolicyConfig")
        .def(py::init<>())
        .def(py::init<ExplorationScore, FinalizationScore, std::string, float, float, float, Int, Int, Int, Int, Int,
                      Int, Int, Int, Int, Int, Int, bool>(),
             py::arg("ExplorationScore") = ExplorationScore(), 
             py::arg("FinalizationScore") = FinalizationScore(),
             py::arg("selection") = "greedy",
             py::arg("temperature") = 0.0f, 
             py::arg("non_improving_prob") = 0.0f,
             py::arg("gen_part") = 1.0f, 
             py::arg("num_samples") = 64, 
             py::arg("num_candidates") = 1,
             py::arg("top_pool") = 1, 
             py::arg("max_from_single_ns") = 10, 
             py::arg("min_reduction") = 0,
             py::arg("max_reduction") = k_single_sentinel<Int>(), 
             py::arg("min_z_to_research") = 1 << 30,
             py::arg("max_z_to_research") = 1 << 30,
             py::arg("min_pool_size") = 1, 
             py::arg("max_tohpe") = 1, 
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
        .def_readwrite("min_z_to_research", &PolicyConfig::min_z_to_research)
        .def_readwrite("max_z_to_research", &PolicyConfig::max_z_to_research)
        .def_readwrite("gen_part", &PolicyConfig::gen_part)
        .def_readwrite("max_tohpe", &PolicyConfig::max_tohpe)
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
                   ", min_z_to_research=" + std::to_string((long long)c.min_z_to_research) +
                   ", gen_part=" + std::to_string((long long)c.gen_part) +
                   ", max_tohpe=" + std::to_string((long long)c.max_tohpe) +
                   ", try_only_tohpe=" + std::to_string((long long)c.try_only_tohpe) +
                   ", tohpe_sample=" + std::to_string((long long)c.tohpe_sample) + ")";
        });

    m.def(
        "policy_iteration",
        [](Matrix cur_mat, PolicyConfig pcfg, index_t seed, index_t add_seed) {
            auto data = std::make_shared<MatrixWithData>(std::move(cur_mat));
            return policy_iteration_impl(data, pcfg, seed, add_seed);
        },
        py::arg("cur_mat"), py::arg("policy_cfg") = PolicyConfig{}, py::arg("seed") = 21, py::arg("add_seed") = 0,
        py::call_guard<py::gil_scoped_release>());
}
