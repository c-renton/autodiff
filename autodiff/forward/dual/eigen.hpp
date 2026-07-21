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

#pragma once

// Eigen includes
#include <Eigen/Core>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/utils/gradient.hpp>
#include <autodiff/common/eigen.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

using namespace autodiff;
using namespace autodiff::detail;

template<typename T>
struct NumTraits;

template<typename T, typename G>
struct NumTraits<autodiff::Dual<T, G>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Dual<T, G> Real;
    typedef autodiff::Dual<T, G> NonInteger;
    typedef autodiff::Dual<T, G> Nested;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<Dual<T, G>, NumericType<T>, BinOp>
{
    typedef DualType<Dual<T, G>> ReturnType;
};

template<typename Op, typename R, typename BinOp>
struct ScalarBinaryOpTraits<UnaryExpr<Op, R>, NumericType<UnaryExpr<Op, R>>, BinOp>
{
    typedef DualType<UnaryExpr<Op, R>> ReturnType;
};

template<typename Op, typename L, typename R, typename BinOp>
struct ScalarBinaryOpTraits<BinaryExpr<Op, L, R>, NumericType<BinaryExpr<Op, L, R>>, BinOp>
{
    typedef DualType<BinaryExpr<Op, L, R>> ReturnType;
};

template<typename Op, typename L, typename C, typename R, typename BinOp>
struct ScalarBinaryOpTraits<TernaryExpr<Op, L, C, R>, NumericType<TernaryExpr<Op, L, C, R>>, BinOp>
{
    typedef DualType<TernaryExpr<Op, L, C, R>> ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<T>, Dual<T, G>, BinOp>
{
    typedef DualType<Dual<T, G>> ReturnType;
};

template<typename Op, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<UnaryExpr<Op, R>>, UnaryExpr<Op, R>, BinOp>
{
    typedef DualType<UnaryExpr<Op, R>> ReturnType;
};

template<typename Op, typename L, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<BinaryExpr<Op, L, R>>, BinaryExpr<Op, L, R>, BinOp>
{
    typedef DualType<BinaryExpr<Op, L, R>> ReturnType;
};

template<typename Op, typename L, typename C, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<TernaryExpr<Op, L, C, R>>, TernaryExpr<Op, L, C, R>, BinOp>
{
    typedef DualType<TernaryExpr<Op, L, C, R>> ReturnType;
};

//------------------------------------------------------------------------------
// SUPPORT FOR Eigen::numext::sqrt (and similar) CALLED DIRECTLY ON UNEVALUATED
// EXPRESSION TEMPLATES (UnaryExpr/BinaryExpr/TernaryExpr).
//
// Since Eigen 5, functions like Eigen::internal::makeHouseholder() call
// `numext::sqrt(numext::abs2(c0) + tailSqNorm)` using a *qualified* call to
// numext::sqrt (previously an unqualified call relying on ADL, which happened
// to pick up autodiff's own lazy `sqrt(R&&)` overload). Because the call is
// qualified, ADL no longer applies, so `Scalar` in `numext::sqrt<Scalar>` gets
// deduced directly from the argument's raw (lazy, unevaluated) expression type,
// e.g. BinaryExpr<AddOp, Dual, Dual&>, rather than the concrete Dual type.
//
// Eigen's numext::sqrt is defined as:
//   template<typename Scalar>
//   EIGEN_MATHFUNC_RETVAL(sqrt, Scalar) sqrt(const Scalar& x)
//   { return EIGEN_MATHFUNC_IMPL(sqrt, Scalar)::run(x); }
// where EIGEN_MATHFUNC_RETVAL(sqrt, Scalar) expands to
// `typename Eigen::internal::sqrt_retval<Scalar>::type`, which defaults to
// `Scalar` itself. For an expression type like BinaryExpr<AddOp, L, R>, that
// return type is not just wrong but *impossible* to construct (sqrt of a sum
// cannot be represented as an AddExpr), causing a hard compile error.
//
// The fix is to specialize both `sqrt_retval` and `sqrt_impl` for the
// expression template types, so that the outer numext::sqrt() call returns
// the concrete evaluated Dual type instead of trying to preserve the
// (unrepresentable) expression type. `autodiff::detail::eval()` performs the
// eager evaluation of the expression down to its concrete Dual value.
//------------------------------------------------------------------------------
namespace internal {

template<typename Op, typename R>
struct sqrt_retval<UnaryExpr<Op, R>>
{
    typedef DualType<UnaryExpr<Op, R>> type;
};

template<typename Op, typename R>
struct sqrt_impl<UnaryExpr<Op, R>>
{
    using Expr   = UnaryExpr<Op, R>;
    using Result = DualType<Expr>;
    static EIGEN_ALWAYS_INLINE Result run(const Expr& x)
    {
        using autodiff::detail::sqrt;
        return sqrt(autodiff::detail::eval(x));
    }
};

template<typename Op, typename L, typename R>
struct sqrt_retval<BinaryExpr<Op, L, R>>
{
    typedef DualType<BinaryExpr<Op, L, R>> type;
};

template<typename Op, typename L, typename R>
struct sqrt_impl<BinaryExpr<Op, L, R>>
{
    using Expr   = BinaryExpr<Op, L, R>;
    using Result = DualType<Expr>;
    static EIGEN_ALWAYS_INLINE Result run(const Expr& x)
    {
        using autodiff::detail::sqrt;
        return sqrt(autodiff::detail::eval(x));
    }
};

template<typename Op, typename L, typename C, typename R>
struct sqrt_retval<TernaryExpr<Op, L, C, R>>
{
    typedef DualType<TernaryExpr<Op, L, C, R>> type;
};

template<typename Op, typename L, typename C, typename R>
struct sqrt_impl<TernaryExpr<Op, L, C, R>>
{
    using Expr   = TernaryExpr<Op, L, C, R>;
    using Result = DualType<Expr>;
    static EIGEN_ALWAYS_INLINE Result run(const Expr& x)
    {
        using autodiff::detail::sqrt;
        return sqrt(autodiff::detail::eval(x));
    }
};

} // namespace internal

} // namespace Eigen

namespace autodiff {

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual0th, dual0th);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual1st, dual1st);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual2nd, dual2nd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual3rd, dual3rd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual4th, dual4th);

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual, dual)

} // namespace autodiff
