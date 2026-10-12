#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>
#include <lis.h>
#ifdef conj
#undef conj
#endif

#include "gmres_2.hpp"

namespace {

struct Result {
  LIS_INT status;
  LIS_INT iterations;
  LIS_REAL reported_residual;
  LIS_REAL relative_residual;
  LIS_REAL solution_error;
};

using SparseMatrix = Eigen::SparseMatrix<double>;
using Vector = Eigen::VectorXd;

Result solve_custom(const SparseMatrix &matrix, const Vector &b,
                    const Vector &exact_solution, int max_iterations,
                    int restart, double tolerance) {
  Vector x = Vector::Zero(b.size());
  Eigen::DiagonalPreconditioner<double> preconditioner(matrix);
  int iterations = max_iterations;
  double reported_residual = tolerance;
  const int status = LinearAlgebra::GMRES(
      matrix, x, b, preconditioner, iterations, restart, reported_residual);
  const double relative_residual = (b - matrix * x).norm() / b.norm();
  return {status, iterations, reported_residual, relative_residual,
          (x - exact_solution).norm()};
}

Result solve_lis(const SparseMatrix &input, const Vector &rhs,
                 const Vector &exact_solution, LIS_INT max_iterations,
                 LIS_INT restart, LIS_REAL tolerance, int argc, char **argv) {
  const LIS_INT n = input.rows();
  LIS_MATRIX matrix = nullptr;
  LIS_VECTOR b = nullptr;
  LIS_VECTOR x = nullptr;
  LIS_SOLVER solver = nullptr;
  bool initialized = false;

  auto cleanup = [&] {
    if (solver != nullptr)
      lis_solver_destroy(solver);
    if (x != nullptr)
      lis_vector_destroy(x);
    if (b != nullptr)
      lis_vector_destroy(b);
    if (matrix != nullptr)
      lis_matrix_destroy(matrix);
    if (initialized)
      lis_finalize();
  };

  auto check = [&](LIS_INT code, const char *operation) {
    if (code != LIS_SUCCESS)
      throw std::runtime_error(std::string(operation) +
                               " failed with LIS error " +
                               std::to_string(code));
  };

  try {
    check(lis_initialize(&argc, &argv), "lis_initialize");
    initialized = true;

    check(lis_matrix_create(LIS_COMM_WORLD, &matrix), "lis_matrix_create");
    check(lis_matrix_set_size(matrix, 0, n), "lis_matrix_set_size");
    for (LIS_INT row = 0; row < n; ++row) {
      for (LIS_INT column = 0; column < n; ++column) {
        const LIS_REAL value = input.coeff(row, column);
        if (value != 0.0)
          check(lis_matrix_set_value(LIS_INS_VALUE, row, column, value, matrix),
                "lis_matrix_set_value");
      }
    }
    check(lis_matrix_assemble(matrix), "lis_matrix_assemble");

    check(lis_vector_create(LIS_COMM_WORLD, &b), "lis_vector_create");
    check(lis_vector_set_size(b, 0, n), "lis_vector_set_size");
    check(lis_vector_create(LIS_COMM_WORLD, &x), "lis_vector_create");
    check(lis_vector_set_size(x, 0, n), "lis_vector_set_size");
    for (LIS_INT row = 0; row < n; ++row) {
      check(lis_vector_set_value(LIS_INS_VALUE, row, rhs(row), b),
            "lis_vector_set_value");
      check(lis_vector_set_value(LIS_INS_VALUE, row, 0.0, x),
            "lis_vector_set_value");
    }

    check(lis_solver_create(&solver), "lis_solver_create");
    std::string options = "-i gmres -p jacobi -maxiter " +
                          std::to_string(max_iterations) + " -restart " +
                          std::to_string(restart) + " -tol " +
                          std::to_string(tolerance);
    check(lis_solver_set_option(options.data(), solver),
          "lis_solver_set_option");
    check(lis_solve(matrix, b, x, solver), "lis_solve");

    LIS_INT iterations = 0;
    LIS_REAL reported_residual = 0.0;
    check(lis_solver_get_iter(solver, &iterations), "lis_solver_get_iter");
    check(lis_solver_get_residualnorm(solver, &reported_residual),
          "lis_solver_get_residualnorm");

    std::vector<LIS_REAL> solution(n);
    check(lis_vector_get_values(x, 0, n, solution.data()),
          "lis_vector_get_values");

    LIS_REAL residual_squared = 0.0;
    LIS_REAL right_hand_side_squared = 0.0;
    LIS_REAL solution_error_squared = 0.0;
    for (LIS_INT row = 0; row < n; ++row) {
      LIS_REAL product = 0.0;
      for (LIS_INT column = 0; column < n; ++column)
        product += input.coeff(row, column) * solution[column];
      const LIS_REAL expected = rhs(row);
      const LIS_REAL residual = expected - product;
      residual_squared += residual * residual;
      right_hand_side_squared += expected * expected;
      const LIS_REAL error = solution[row] - exact_solution(row);
      solution_error_squared += error * error;
    }

    Result result{LIS_SUCCESS, iterations, reported_residual,
                  std::sqrt(residual_squared / right_hand_side_squared),
                  std::sqrt(solution_error_squared)};
    cleanup();
    return result;
  } catch (...) {
    cleanup();
    throw;
  }
}

void print_result(const Result &result) {
  std::cout << "LIS GMRES"
            << " status=" << result.status
            << " iterations=" << result.iterations
            << " reported residual=" << std::scientific
            << result.reported_residual
            << " measured residual=" << result.relative_residual
            << " solution error=" << result.solution_error << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    constexpr LIS_INT n = 420;
    constexpr LIS_INT max_iterations = 3000;
    constexpr LIS_INT restart = 30;
    constexpr LIS_REAL tolerance = 1.0e-16;

    SparseMatrix matrix;
    if (!Eigen::loadMarket(matrix, "Atest.mtx"))
      throw std::runtime_error("failed to load Atest.mtx");
    if (matrix.rows() != n || matrix.cols() != n)
      throw std::runtime_error("Atest.mtx has unexpected dimensions");
    std::cout << "GMRES comparison for Atest.mtx\n"
              << "matrix: " << matrix.rows() << " x " << matrix.cols()
              << ", nonzeros: " << matrix.nonZeros() << '\n'
              << "restart: " << restart
              << ", maximum iterations: " << max_iterations
              << ", tolerance: " << tolerance << "\n\n";
    const Vector exact_solution = Vector::Ones(n);
    const Vector b = matrix * exact_solution;

    const Result custom =
        solve_custom(matrix, b, exact_solution, max_iterations, restart,
                     tolerance);
    std::cout << "Custom GMRES (gmres_2.hpp)"
              << " status=" << custom.status
              << " iterations=" << custom.iterations
              << " reported residual=" << std::scientific
              << custom.reported_residual
              << " measured residual=" << custom.relative_residual
              << " solution error=" << custom.solution_error << '\n';

    const Result lis = solve_lis(matrix, b, exact_solution, max_iterations,
                                 restart, tolerance, argc, argv);
    print_result(lis);

    std::cout << "\nComparison (custom - LIS): iterations="
              << (custom.iterations - lis.iterations)
              << ", measured residual difference="
              << (custom.relative_residual - lis.relative_residual)
              << ", solution error difference="
              << (custom.solution_error - lis.solution_error) << '\n';
  } catch (const std::exception &error) {
    std::cerr << "LIS GMRES failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
