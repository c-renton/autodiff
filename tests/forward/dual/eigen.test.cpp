//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Eigen includes
#include <Eigen/Geometry>
#include <Eigen/QR>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Catch::Approx
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Catch::Approx(val(std::forward<T>(expr))).margin(zero);
}

#define CHECK_APPROX(a, b) CHECK( abs(a - b) < abs(b) * std::numeric_limits<double>::epsilon() * 100 );

TEST_CASE("testing autodiff::dual (with eigen)", "[forward][dual][eigen]")
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Eigen::ArrayXd;

    SECTION("testing array-unpacking of derivatives for eigen vector of dual numbers")
    {
        dual4th x;
        detail::seed<0>(x, 2.0);
        detail::seed<1>(x, 3.0);
        detail::seed<2>(x, 4.0);
        detail::seed<3>(x, 5.0);
        detail::seed<4>(x, 6.0);

        dual4th y;
        detail::seed<0>(y, 3.0);
        detail::seed<1>(y, 4.0);
        detail::seed<2>(y, 5.0);
        detail::seed<3>(y, 6.0);
        detail::seed<4>(y, 7.0);

        dual4th z;
        detail::seed<0>(z, 4.0);
        detail::seed<1>(z, 5.0);
        detail::seed<2>(z, 6.0);
        detail::seed<3>(z, 7.0);
        detail::seed<4>(z, 8.0);

        VectorXdual4th u(3);
        u << x, y, z;

        auto [u0, u1, u2, u3, u4] = derivatives(u);

        CHECK( u0[0] == approx(derivative<0>(x)) );
        CHECK( u0[1] == approx(derivative<0>(y)) );
        CHECK( u0[2] == approx(derivative<0>(z)) );

        CHECK( u1[0] == approx(derivative<1>(x)) );
        CHECK( u1[1] == approx(derivative<1>(y)) );
        CHECK( u1[2] == approx(derivative<1>(z)) );

        CHECK( u2[0] == approx(derivative<2>(x)) );
        CHECK( u2[1] == approx(derivative<2>(y)) );
        CHECK( u2[2] == approx(derivative<2>(z)) );

        CHECK( u3[0] == approx(derivative<3>(x)) );
        CHECK( u3[1] == approx(derivative<3>(y)) );
        CHECK( u3[2] == approx(derivative<3>(z)) );

        CHECK( u4[0] == approx(derivative<4>(x)) );
        CHECK( u4[1] == approx(derivative<4>(y)) );
        CHECK( u4[2] == approx(derivative<4>(z)) );
    }

    SECTION("testing casting to VectorXd")
    {
        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXd y = x.cast<double>();

        for(auto i = 0; i < 3; ++i)
            CHECK_APPROX( x[i], y(i) );
    }

    SECTION("testing casting to MatriXd")
    {
        MatrixXdual x(2, 2);
        x << 0.5, 0.2, 0.3, 0.7;

        MatrixXd y = x.cast<double>();

        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                CHECK_APPROX( x(i, j), y(i, j) );
    }

    SECTION("testing multiplication of VectorXdual by MatrixXd")
    {
        MatrixXd A(2, 2);
        A << 1.0, 3.0, 5.0, 7.0;

        VectorXdual x(2);

        detail::seed<0>(x[0], 1.0);
        detail::seed<0>(x[1], 2.0);

        detail::seed<1>(x[0], 3.0);
        detail::seed<1>(x[1], 5.0);

        VectorXdual b = A * x;

        CHECK( derivative<0>(b[0]) == approx(7.0) );
        CHECK( derivative<0>(b[1]) == approx(19.0) );

        CHECK( derivative<1>(b[0]) == approx(18.0) );
        CHECK( derivative<1>(b[1]) == approx(50.0) );
    }

    SECTION("testing class template specializations for Eigen::ScalarBinaryOpTraits")
    {
        dual4th x = 4.0;

        MatrixXd A(2, 2);
        A << 1.0, 3.0, 5.0, 7.0;

        // The checks below also allow us to determine if compilation succeeds.
        // Note that we mix not only dual4th times MatrixXd, but also expressions,
        // such as UnaryExp -x and +x, BinaryExpr (x+x) and (x+x)*(x+x).

        CHECK( (x*A).isApprox(x*A.cast<dual4th>()) );
        CHECK( ((+x)*A).isApprox(+x*A.cast<dual4th>()) );
        CHECK( ((-x)*A).isApprox(-x*A.cast<dual4th>()) );
        CHECK( ((x+x)*A).isApprox(2*x*A.cast<dual4th>()) );
        CHECK( (((x+x)*(x+x))*A).isApprox(4*x*x*A.cast<dual4th>()) );

        CHECK( (A*x).isApprox(x*A.cast<dual4th>()) );
        CHECK( (A*(+x)).isApprox(+x*A.cast<dual4th>()) );
        CHECK( (A*(-x)).isApprox(-x*A.cast<dual4th>()) );
        CHECK( (A*(x+x)).isApprox(2*x*A.cast<dual4th>()) );
        CHECK( (A*((x+x)*(x+x))).isApprox(4*x*x*A.cast<dual4th>()) );
    }

    SECTION("using Eigen::VectorXdual")
    {
        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorXdual& x) -> dual
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing gradient derivatives of only the last two variables")
        {
            auto f = [](const VectorXdual& x) -> dual
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd g = gradient(f, wrt(x.tail(2)), at(x));

            CHECK( g[0] == approx(x[1]) );
            CHECK( g[1] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorXdual& x) -> VectorXdual
            {
                return x / x.array().sum();
            };

            VectorXdual x(3);
            x << 0.5, 0.2, 0.3;

            VectorXdual F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing jacobian derivatives of only the last two variables")
        {
            auto f = [](const VectorXdual& x) -> VectorXdual
            {
                return x / x.array().sum();
            };

            VectorXdual x(3);
            x << 0.5, 0.2, 0.3;

            VectorXdual F;

            const MatrixXd J = jacobian(f, wrt(x.tail(2)), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 2; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j + 1) ? 1.0 : 0.0)) );
        }

        SECTION("testing casting to VectorXd")
        {
            VectorXdual x(3);
            x << 0.5, 0.2, 0.3;

            VectorXd y = x.cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK_APPROX( x(i), y(i) );
        }

        SECTION("testing casting to VectorXf")
        {
            MatrixXdual x(2, 2);
            x << 0.5, 0.2, 0.3, 0.7;
            MatrixXd y = x.cast<double>();
            for(auto i = 0; i < 2; ++i)
                for(auto j = 0; j < 2; ++j)
                    CHECK_APPROX( x(i, j), y(i, j) );
        }

        SECTION("test gradient size with respect to few arguments")
        {
            auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> dual
            {
                return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
            };

            VectorXdual x(3);
            x << 1.0, 2.0, 3.0;

            dual y = 2;

            VectorXdual z(4);
            z << 1.0, 2.0, 3.0, 4.0;

            VectorXd g = gradient(f, wrt(x.tail(2), y, z), at(x, y, z));

            CHECK( g.size() == 7 );
        }

        SECTION("testing gradient derivatives wrt pack variables")
        {
            auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> dual
            {
                return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
            };

            VectorXdual x(2);
            x << 1.0, 2.0;

            dual y = 3.0;

            VectorXdual z(1);
            z << 4.0;

            VectorXd g = gradient(f, wrt(x, y, z), at(x, y, z));

            CHECK(g[0] == approx(x[0]));
            CHECK(g[1] == approx(x[1]));
            CHECK(g[2] == approx(y));
            CHECK(g[3] == approx(z[0]));
        }

        SECTION("test jacobian size with respect to few arguments")
        {
            auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> VectorXdual
            {
                VectorXdual ret(x.size() + z.size());
                ret.head(x.size()) = x * y / x.array().sum();
                ret.tail(z.size()) = y * z;

                return ret;
            };

            VectorXdual x(3);
            x << 0.5, 0.2, 0.3;

            dual y = 2.0;

            VectorXdual z(3);
            z << 1.0, 2.0, 3.0;

            VectorXdual F;

            const MatrixXd J = jacobian(f, wrt(x.tail(2), y), at(x, y, z), F);

            CHECK(J.rows() == 6);
            CHECK(J.cols() == 3);
        }

        SECTION("test jacobian size with respect to few arguments")
        {
            auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> VectorXdual
            {
                VectorXdual ret(x.size() + z.size());
                ret.head(x.size()) = x * y / x.array().sum();
                ret.tail(z.size()) = y * z;

                return ret;
            };

            VectorXdual x(3);
            x << 0.5, 0.2, 0.3;

            dual y = 2.0;

            VectorXdual z(3);
            z << 1.0, 2.0, 3.0;

            VectorXdual F;

            const MatrixXd J = jacobian(f, wrt(x, y, z), at(x, y, z), F);

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i, j) == approx(-F[i] + ((i == j) ? y.val : 0.0)));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i + 3, j) == approx(0.0));

            for (auto i = 0; i < 6; ++i)
                    CHECK(J(i, 3) == approx( i < 3 ? x(i) : z(i - 3)));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i, j + 4) == approx(0.0));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i + 3, j + 4) == approx((i == j) ? y.val : 0.0));
        }
    }

    SECTION("using VectorXdual2nd")
    {
        SECTION("testing casting to VectorXd")
        {
            VectorXdual2nd x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd y = x.template cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK_APPROX( x(i), y(i) );
        }

        using dual2nd = HigherOrderDual<2, double>;

        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorXdual2nd& x) -> dual2nd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual2nd x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorXdual2nd& x) -> VectorXdual2nd
            {
                return x / x.array().sum();
            };

            VectorXdual2nd x(3);
            x << 0.5, 0.2, 0.3;

            VectorXdual2nd F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing hessian derivatives")
        {
            auto f = [](const VectorXdual2nd& x) -> dual2nd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual2nd x(3);
            x << 1.0, 2.0, 3.0;

            MatrixXd H = hessian(f, wrt(x), at(x));

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( H(i, j) == approx(((i == j) ? 1.0 : 0.0)) );
        }
    }

    SECTION("using VectorXdual3rd")
    {
        using dual3rd = HigherOrderDual<2, double>;
        using VectorXdual3rd = Eigen::Matrix<dual3rd, -1, 1, 0, -1, 1>;

        SECTION("testing casting to VectorXd")
        {
            VectorXdual3rd x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd y = x.template cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK_APPROX( x(i), y(i) );
        }

        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorXdual3rd& x) -> dual3rd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual3rd x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorXdual3rd& x) -> VectorXdual3rd
            {
                return x / x.array().sum();
            };

            VectorXdual3rd x(3);
            x << 0.5, 0.2, 0.3;

            VectorXdual3rd F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing hessian derivatives")
        {
            auto f = [](const VectorXdual3rd& x) -> dual3rd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual3rd x(3);
            x << 1.0, 2.0, 3.0;

            MatrixXd H = hessian(f, wrt(x), at(x));

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( H(i, j) == approx(((i == j) ? 1.0 : 0.0)) );
        }
    }

    SECTION("testing Eigen::HouseholderQR with autodiff::dual (regression test for numext::sqrt on lazy expression templates)")
    {
        // Eigen >= 5 changed Eigen::internal::makeHouseholder() to compute
        //     beta = numext::sqrt(numext::abs2(c0) + tailSqNorm);
        // using a *qualified* call to numext::sqrt (previously an unqualified
        // call relying on ADL to find autodiff's own lazy sqrt() overload).
        // Because dual arithmetic produces lazy expression templates
        // (UnaryExpr/BinaryExpr/TernaryExpr) instead of eagerly evaluating to
        // `dual`, the qualified call causes Eigen::internal::sqrt_impl<Scalar>
        // to be instantiated with Scalar deduced as the raw, unevaluated
        // expression type, which used to fail to compile (sqrt of a sum cannot
        // be represented as that same sum expression). This is a regression
        // test for the Eigen::internal::sqrt_retval/sqrt_impl specializations
        // added above for UnaryExpr/BinaryExpr/TernaryExpr.
        auto qrTopLeft = [](const VectorXdual& p) -> dual
        {
            MatrixXdual A(2, 2);
            A << p[0], p[1],
                 p[1], p[0] + 2.0;

            Eigen::HouseholderQR<Eigen::Ref<MatrixXdual>> qr(A); // In-place QR decomposition; used to fail to compile
            MatrixXdual R = A.template triangularView<Eigen::Upper>();
            return R(0, 0);
        };

        auto qrTopLeftDouble = [](const VectorXd& p) -> double
        {
            MatrixXd A(2, 2);
            A << p[0], p[1],
                 p[1], p[0] + 2.0;

            Eigen::HouseholderQR<MatrixXd> qr(A);
            MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>();
            return R(0, 0);
        };

        VectorXdual p(2);
        p << 4.0, 1.0;

        dual F;
        VectorXd g = gradient(qrTopLeft, wrt(p), at(p), F);

        VectorXd pd(2);
        pd << 4.0, 1.0;

        CHECK( F == approx(qrTopLeftDouble(pd)) );

        // For this symmetric 2x2 matrix, the Householder reflection reduces
        // the first column [p0, p1] to a single value: since c0 = p0 >= 0,
        // beta is negated, giving R(0,0) = -sqrt(p0^2 + p1^2) exactly (the
        // signed column norm), with gradient d/dp = -p / sqrt(p0^2 + p1^2).
        const double norm = std::sqrt(pd[0]*pd[0] + pd[1]*pd[1]);
        CHECK( F == approx(-norm) );
        CHECK( g[0] == Catch::Approx(-pd[0] / norm) );
        CHECK( g[1] == Catch::Approx(-pd[1] / norm) );
    }

    SECTION("testing Eigen::Quaternion<dual>::slerp() (minimal regression test for numext::sqrt on lazy expression templates)")
    {
        // A more minimal reproduction of the same numext::sqrt regression
        // demonstrated above via Eigen::HouseholderQR. QuaternionBase::slerp()
        // computes
        //     Scalar sinTheta = numext::sqrt(Scalar(1) - absD * absD);
        // using a *qualified* call to numext::sqrt on an inline expression
        // (absD * absD, then Scalar(1) - ...) rather than a stored variable.
        // For dual operands this produces nested lazy UnaryExpr/BinaryExpr
        // types rather than an eagerly-evaluated dual, so Scalar in
        // numext::sqrt<Scalar> gets deduced as that raw expression type,
        // which used to fail to compile without the sqrt_retval/sqrt_impl
        // specializations added above. A single slerp() call is enough to
        // trigger it -- no matrix decomposition required.
        auto slerpW = [](const VectorXdual& p) -> dual
        {
            Eigen::Quaternion<dual> q1(dual(1.0), dual(0.0), dual(0.0), dual(0.0));
            Eigen::Quaternion<dual> q2(p[0], p[1], dual(0.0), dual(0.0));
            Eigen::Quaternion<dual> q3 = q1.slerp(dual(0.5), q2); // used to fail to compile
            return q3.w();
        };

        auto slerpWDouble = [](const VectorXd& p) -> double
        {
            Eigen::Quaterniond q1(1.0, 0.0, 0.0, 0.0);
            Eigen::Quaterniond q2(p[0], p[1], 0.0, 0.0);
            Eigen::Quaterniond q3 = q1.slerp(0.5, q2);
            return q3.w();
        };

        VectorXdual p(2);
        p << 0.7071, 0.7071;

        dual F;
        VectorXd g = gradient(slerpW, wrt(p), at(p), F);

        VectorXd pd(2);
        pd << 0.7071, 0.7071;

        const double fval = slerpWDouble(pd);
        CHECK( F == approx(fval) );

        // Since q1 is the identity quaternion, d = q1.dot(q2) = q1.w()*q2.w()
        // = p0 depends only on p0 (q1's zero x/y/z coefficients annihilate the
        // p1 term), so theta = acos(p0), scale0, and scale1 all depend only on
        // p0. With t = 0.5, scale0 = scale1 = sin(theta/2)/sin(theta), and
        // F = scale0 + scale1*p0 simplifies via the half-angle identities
        // sin(theta/2)/sin(theta) = 1/(2*cos(theta/2)) and
        // cos(theta/2) = sqrt((1+p0)/2) to the closed form:
        //     F = sqrt((1+p0)/2) = cos(theta/2)
        // (the well-known slerp-midpoint identity: halfway between the
        // identity rotation and a rotation by theta is a rotation by
        // theta/2). Hence dF/dp0 = 1/(4*F) and dF/dp1 = 0 (p1 never enters
        // the computation for this q1).
        CHECK( fval == Catch::Approx(std::sqrt((1.0 + pd[0]) / 2.0)) );
        CHECK( g[0] == Catch::Approx(1.0 / (4.0 * fval)) );
        CHECK( g[1] == Catch::Approx(0.0).margin(1.0e-12) );
    }

    SECTION("testing VectorXdual::stableNorm() (regression test for numext::abs2 on lazy expression templates)")
    {
        // Eigen >= 5 (and also 3.4, at this call site) computes stableNorm() via
        // Eigen::internal::stable_norm_kernel(), which contains:
        //     ssq = ssq * numext::abs2(scale / maxCoeff);
        // using a *qualified* call to numext::abs2. Because (scale / maxCoeff)
        // for dual operands produces a lazy BinaryExpr<MulOp, dual&,
        // UnaryExpr<InvOp, dual&>> rather than an eagerly-evaluated dual,
        // abs2_impl<Scalar> is instantiated with Scalar deduced as that raw
        // expression type. The default abs2_impl_default::run() computes x*x,
        // which deepens the expression tree to BinaryExpr<MulOp, BinaryExpr<...>,
        // BinaryExpr<...>> -- a type that cannot convert back to the original
        // BinaryExpr<MulOp,...> declared as the RealScalar return type, causing
        // a hard compile error.
        // This is the same class of bug as the numext::sqrt / HouseholderQR issue
        // above (qualified numext:: call bypasses ADL so the lazy expression type
        // is deduced as Scalar directly), but in abs2_impl rather than sqrt_impl.
        // Fix: add Eigen::internal::abs2_retval / abs2_impl specializations for
        // UnaryExpr/BinaryExpr/TernaryExpr, analogous to the sqrt fix in eigen.hpp.
        auto stableNormFn = [](const VectorXdual& x) -> dual
        {
            return x.stableNorm(); // triggers numext::abs2(scale / maxCoeff) in StableNorm.h
        };

        auto stableNormDouble = [](const VectorXd& x) -> double
        {
            return x.stableNorm();
        };

        VectorXdual x(3);
        x << 3.0, 4.0, 0.0;

        dual F;
        VectorXd g = gradient(stableNormFn, wrt(x), at(x), F);

        VectorXd xd(3);
        xd << 3.0, 4.0, 0.0;

        CHECK( F == approx(stableNormDouble(xd)) );

        // The gradient of ||x|| is x / ||x||.
        const double norm = stableNormDouble(xd);
        for (auto i = 0; i < 3; ++i)
            CHECK( g[i] == Catch::Approx(xd[i] / norm) );
    }

    SECTION("testing numext::sqrt directly on BinaryExpr<AddOp> (a + b)")
    {
        // Directly exercises the sqrt_retval/sqrt_impl specialisation for
        // BinaryExpr added to eigen.hpp, independently of any particular Eigen
        // algorithm's internals. Models the expression shape arising in
        // Householder.h (numext::sqrt(numext::abs2(c0) + tailSqNorm)) and
        // Quaternion.h (numext::sqrt(Scalar(1) - absD * absD)).
        // Fails to compile without the sqrt_retval/sqrt_impl specialisations.
        auto f = [](const VectorXdual& p) -> dual
        {
            return Eigen::numext::sqrt(p[0] + p[1]);
        };
        auto fref = [](const VectorXd& p) -> double
        {
            return std::sqrt(p[0] + p[1]);
        };

        VectorXdual p(2); p << 4.0, 5.0;
        VectorXd pd(2);   pd << 4.0, 5.0;

        dual F;
        VectorXd g = gradient(f, wrt(p), at(p), F);

        const double fval = fref(pd);
        CHECK( F == approx(fval) );

        // f(a,b) = sqrt(a+b); df/da = df/db = 1 / (2*sqrt(a+b))
        const double dfd = 1.0 / (2.0 * fval);
        CHECK( g[0] == Catch::Approx(dfd) );
        CHECK( g[1] == Catch::Approx(dfd) );
    }

    SECTION("testing numext::sqrt directly on BinaryExpr<MulOp/InvOp> (a / b)")
    {
        // Directly exercises the sqrt_retval/sqrt_impl specialisation for
        // BinaryExpr. Division a/b produces BinaryExpr<MulOp, dual&,
        // UnaryExpr<InvOp, dual&>>, modelling the expression shape in
        // ConjugateGradient.h (numext::sqrt(residualNorm2 / rhsNorm2)).
        // Fails to compile without the sqrt_retval/sqrt_impl specialisations.
        auto f = [](const VectorXdual& p) -> dual
        {
            return Eigen::numext::sqrt(p[0] / p[1]);
        };
        auto fref = [](const VectorXd& p) -> double
        {
            return std::sqrt(p[0] / p[1]);
        };

        VectorXdual p(2); p << 9.0, 1.0;
        VectorXd pd(2);   pd << 9.0, 1.0;

        dual F;
        VectorXd g = gradient(f, wrt(p), at(p), F);

        const double fval = fref(pd);
        CHECK( F == approx(fval) );

        // f(a,b) = sqrt(a/b); df/da = 1/(2*b*sqrt(a/b)); df/db = -a/(2*b^2*sqrt(a/b))
        CHECK( g[0] == Catch::Approx(1.0 / (2.0 * pd[1] * fval)) );
        CHECK( g[1] == Catch::Approx(-pd[0] / (2.0 * pd[1] * pd[1] * fval)) );
    }

    SECTION("testing numext::abs2 directly on BinaryExpr<MulOp/InvOp> (a / b)")
    {
        // Directly exercises the abs2_retval/abs2_impl specialisation for
        // BinaryExpr, modelling the expression shape in StableNorm.h
        // (numext::abs2(scale / maxCoeff)).
        // Fails to compile without the abs2_retval/abs2_impl specialisations.
        auto f = [](const VectorXdual& p) -> dual
        {
            return Eigen::numext::abs2(p[0] / p[1]);
        };
        auto fref = [](const VectorXd& p) -> double
        {
            return (p[0] / p[1]) * (p[0] / p[1]);
        };

        VectorXdual p(2); p << 6.0, 2.0;
        VectorXd pd(2);   pd << 6.0, 2.0;

        dual F;
        VectorXd g = gradient(f, wrt(p), at(p), F);

        CHECK( F == approx(fref(pd)) );

        // f(a,b) = (a/b)^2; df/da = 2*a/b^2; df/db = -2*a^2/b^3
        CHECK( g[0] == Catch::Approx(2.0 * pd[0] / (pd[1]*pd[1])) );
        CHECK( g[1] == Catch::Approx(-2.0 * pd[0]*pd[0] / (pd[1]*pd[1]*pd[1])) );
    }

    SECTION("testing numext::abs2 directly on UnaryExpr<NegOp> (-a)")
    {
        // Directly exercises the abs2_retval/abs2_impl specialisation for
        // UnaryExpr. Unary negation -p[0] produces UnaryExpr<NegOp, dual&>.
        // abs2(-x) = x^2, so the value and gradient are identical to abs2(x).
        // Fails to compile without the abs2_retval/abs2_impl specialisations.
        auto f = [](const VectorXdual& p) -> dual
        {
            return Eigen::numext::abs2(-p[0]);
        };
        auto fref = [](const VectorXd& p) -> double
        {
            return p[0] * p[0];
        };

        VectorXdual p(1); p << 3.0;
        VectorXd pd(1);   pd << 3.0;

        dual F;
        VectorXd g = gradient(f, wrt(p), at(p), F);

        CHECK( F == approx(fref(pd)) );

        // f(a) = a^2; df/da = 2*a
        CHECK( g[0] == Catch::Approx(2.0 * pd[0]) );
    }

    SECTION("using Eigen::Map")
    {
        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorXdual& x) -> dual
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXdual vec(3);
            vec << 1.0, 2.0, 3.0;

            auto x = Eigen::Map<VectorXdual>(vec.data(), vec.size());

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorXdual& x) -> VectorXdual
            {
                return x / x.array().sum();
            };


            VectorXdual vec(3);
            vec << 0.5, 0.2, 0.3;

            auto x = Eigen::Map<VectorXdual>(vec.data(), vec.size());

            VectorXdual F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }
    }
}
