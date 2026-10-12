

#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Dense>
namespace LinearAlgebra {

inline void GeneratePlaneRotation(double &dx, double &dy, double &cs,
                                  double &sn) {
  if (dy == 0.0) {
    cs = 1.0;
    sn = 0.0;
  } else if (fabs(dy) > fabs(dx)) {
    double temp = dx / dy;
    sn = 1.0 / sqrt(1.0 + temp * temp);
    cs = temp * sn;
  } else {
    double temp = dy / dx;
    cs = 1.0 / sqrt(1.0 + temp * temp);
    sn = temp * cs;
  }
}

inline void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn) {
  double temp = cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = temp;
}

template <class Vector, class Hessenberg>
inline void Update(Vector &x, int k, Hessenberg &h, Vector &s,
                   std::vector<Vector> &v) {
  Vector y(s.size());
  y = s;

  // Backsolve:
  for (int i = k; i >= 0; i--) {
    y(i) /= h(i, i);
    for (int j = i - 1; j >= 0; j--)
      y(j) -= h(j, i) * y(i);
  }

  for (int j = 0; j <= k; j++)
    x += y(j) * v[j];
}

template <class Matrix, class Vector, class Preconditioner>
int GMRES(const Matrix &A, Vector &x, const Vector &b, const Preconditioner &M,
          int &max_iter, int m, typename Vector::Scalar &tol) {

  using Real = typename Matrix::Scalar;

  int n = A.size();

  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> H =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Zero(m + 1, m);
  Vector s(m + 1), cs(m + 1), sn(m + 1);
  Vector w(n), av(n);
  Real resid;
  int i, j, k;

  w = M.solve(b);
  Real normb = w.norm(); // normb = ||M^-1 b||
  if (normb == 0.0)
    normb = 1;

  Vector r(n);
  r = M.solve(b - A * x); // r = M^-1 (b - A x)
  Real beta = r.norm(); // beta = ||r||

  resid = beta / normb;
  if (resid <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }
  j = 1;
  std::vector<Vector> v(m + 1);
  for (i = 0; i <= m; i++)
    v[i] = Vector(n);

  while (j <= max_iter) {

    v[0] = r / beta; // v[0] = r / ||r||
    s.setZero();
    s(0) = beta;
    for (i = 0; i < m && j <= max_iter; i++, j++) {
      av = A * v[i];
      w = M.solve(av); // w = M^1( A * v[i])

      for (k = 0; k <= i; k++) {
        H(k, i) = w.dot(v[k]); // H(k,i) = w * v[k]
        w -= H(k, i) * v[k];   // w -= H(k,i) * v[k]
      }

      H(i + 1, i) = w.norm(); // H(i+1,i) = ||w||
      v[i + 1].setZero();
      if (H(i + 1, i) <= std::numeric_limits<Real>::epsilon()) {
        Update(x, i, H, s, v);
        r = M.solve(b - A * x);
        resid = r.norm() / normb;
        tol = resid;
        max_iter = j;
        return resid <= tol ? 0 : 1;
      }
      v[i + 1] = w / H(i + 1, i); // v[i+1] = w / H(i+1,i)

      for (k = 0; k < i; k++)
        ApplyPlaneRotation(H(k, i), H(k + 1, i), cs(k), sn(k));

      GeneratePlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
      ApplyPlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
      ApplyPlaneRotation(s(i), s(i + 1), cs(i), sn(i));

      resid = std::abs(s(i + 1)) / normb;

      if (resid <= tol) {
        Update(x, i, H, s, v);
        tol = resid;
        max_iter = j;
        return 0;
      }
    }

    Update(x, i - 1, H, s, v);

    w = b - A * x;
    r = M.solve(w);  // r = M^-1 (b - A x)
    beta = r.norm(); // beta = ||r||
    resid = beta / normb;
    if (resid <= tol) {
      tol = resid;
      max_iter = j - 1;
      return 0;
    }
  }
  tol = resid;
  return 1;
}

} // namespace LinearAlgebra